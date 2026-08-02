import os
import re
import time
import concurrent.futures
from pydantic import BaseModel, Field
import json
import io
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types
from PIL import Image
from PIL.ExifTags import TAGS

# Real Railway service and confirmed the actual mechanism: a stalled Gemini
# response (not a clean error, just never returning) blocks this ENTIRE
# service, since process_group() runs groups one at a time in a single
# sequential loop. Nothing else in the queue -- no matter how new -- gets
# picked up until the stuck call finally resolves. Confirmed via real data:
# a group stuck since 15:47 blocked every single scan after it for the rest
# of the day. This wraps any Gemini call in a hard timeout, enforced by this
# code regardless of what timeout (if any) the SDK's own HTTP client uses
# internally, so a stall can never freeze the whole worker again.
_gemini_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def call_gemini_with_timeout(fn, timeout_seconds=90):
    future = _gemini_pool.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise Exception(f"Gemini call did not respond within {timeout_seconds}s (stalled, not a clean error)")

# ------------------------------------------------------------------ #
#  SETUP
# ------------------------------------------------------------------ #

load_dotenv()
url         = os.getenv("SUPABASE_URL")
key         = os.getenv("SUPABASE_KEY")
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "")
api_key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY")

if not all([url, key, api_key]):
    print(f"❌ ERROR: Missing credentials.")
    exit()

supabase = create_client(url, key)
client   = genai.Client(api_key=api_key)

# ------------------------------------------------------------------ #
#  SEEN FILES — stored in Supabase, persists across Railway restarts
# ------------------------------------------------------------------ #

def load_seen() -> set:
    try:
        result = supabase.table("seen_files").select("filename").execute()
        return {row["filename"] for row in (result.data or [])}
    except Exception as _err:
        print(f"⚠️  Could not load seen_files from Supabase: {_err}")
        return set()

def mark_seen(filename: str):
    try:
        supabase.table("seen_files").upsert(
            {"filename": filename},
            on_conflict="filename"
        ).execute()
    except Exception as _err:
        print(f"⚠️  Could not mark {filename} as seen: {_err}")

# ------------------------------------------------------------------ #
#  MODEL PICKER
# ------------------------------------------------------------------ #

def resolve_model():
    print("🔍 Finding best available model...")
    try:
        all_models = [m.name for m in client.models.list()]
        gen_models = [m for m in all_models if "gemini" in m.lower()]
        print(f"Available models: {gen_models}")
        preferred = [
            "gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite",
            "gemini-1.5-flash", "gemini-1.5-pro"
        ]
        for pref in preferred:
            match = next((m for m in gen_models if pref in m), None)
            if match:
                print(f"✅ Using model: {match}")
                return match
        if gen_models:
            print(f"✅ Using model: {gen_models[0]}")
            return gen_models[0]
    except Exception as _err:
        print(f"⚠️  Model list failed: {_err}")
    return "models/gemini-1.5-pro"

model = resolve_model()

# ------------------------------------------------------------------ #
#  EXIF DATE
# ------------------------------------------------------------------ #

def get_exif_date(raw_bytes: bytes):
    try:
        img  = Image.open(io.BytesIO(raw_bytes))
        exif = img._getexif()
        if exif:
            exif_data = {TAGS.get(tag, tag): val for tag, val in exif.items()}
            for field in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                raw_date = exif_data.get(field)
                if raw_date:
                    try:
                        dt = datetime.strptime(str(raw_date), "%Y:%m:%d %H:%M:%S")
                        return dt, dt.isoformat()
                    except ValueError:
                        continue
    except Exception:
        pass
    dt = datetime.now()
    return dt, dt.isoformat()

# ------------------------------------------------------------------ #
#  IMAGE HELPERS
# ------------------------------------------------------------------ #

def to_jpeg_bytes(raw_bytes: bytes) -> bytes:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    from PIL import ImageOps
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

def build_new_filename(dt: datetime, original_name: str) -> str:
    date_code = dt.strftime("%d%m%y")
    time_code = datetime.now().strftime("%H%M%S")
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".heic"):
        ext = ".jpg"
    return f"{date_code}_{time_code}{ext}"

def rename_in_supabase(raw_bytes: bytes, old_name: str, new_name: str) -> bool:
    try:
        supabase.storage.from_("part-photos").upload(
            path=new_name,
            file=raw_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        supabase.storage.from_("part-photos").remove([old_name])
        print(f"   🔁 Renamed: {old_name} → {new_name}")
        return True
    except Exception as _err:
        print(f"   ⚠️  Rename failed: {_err}")
        return False

# ------------------------------------------------------------------ #
#  NUMBER PARSING
# ------------------------------------------------------------------ #

def parse_num(val):
    cleaned = re.sub(r"[^0-9.]", "", str(val))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return 0.00

def parse_int(val):
    cleaned = re.sub(r"[^0-9]", "", str(val))
    try:
        return int(cleaned)
    except ValueError:
        return 0

# ------------------------------------------------------------------ #
#  EBAY FINDING API — SOLD + ACTIVE LISTINGS
# ------------------------------------------------------------------ #

import requests as _requests

def _ebay_find(operation: str, keywords: str, extra_params: dict = {}) -> list[dict]:
    """
    Call eBay Browse API and return list of item dicts with price + title + url.
    operation: findCompletedItems | findItemsAdvanced (kept as labels only -- both
    now hit the Browse API's item_summary/search; it has no true sold-history
    endpoint, that requires eBay's separately-gated Marketplace Insights API).

    PORTED 2026-07-25 from origin/dev commit 4dbaf7c ("fix: replace deprecated eBay
    Finding API with Browse API", May 2026). That fix only ever existed on the dev
    branch. Railway's production branch was switched from dev to main to fix an
    unrelated category-matching bug, which silently reverted pricing back to this
    old, increasingly-503-prone Finding API (svcs.ebay.com) -- main never had this
    fix. Porting it so main has both fixes.
    """
    if not EBAY_APP_ID:
        return []
    token = get_ebay_token()
    if not token:
        return []
    short_kw = " ".join(keywords.split()[:7])
    try:
        import requests as _req
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        }
        params = {"q": short_kw, "limit": "20"}
        if operation == "findCompletedItems":
            params["filter"] = "buyingOptions:{FIXED_PRICE|AUCTION}"
        else:
            params["filter"] = "buyingOptions:{FIXED_PRICE}"
        resp = _req.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            params=params,
            headers=headers,
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("itemSummaries", [])
        results = []
        for item in items:
            try:
                price = float((item.get("price") or {}).get("value", 0))
                if price <= 0:
                    continue
                results.append({
                    "price": price,
                    "title": item.get("title", ""),
                    "url": item.get("itemWebUrl", ""),
                    "condition": item.get("condition") or "Unknown",
                })
            except Exception:
                continue
        print(f"   eBay Browse API ({operation}): {len(results)} results")
        return results
    except Exception as _err:
        print(f"   eBay API error ({operation}): {_err}")
        return []





_ebay_oauth_token = None
_ebay_oauth_expiry = 0

def get_ebay_token():
    global _ebay_oauth_token, _ebay_oauth_expiry
    import time, base64
    import requests as _req
    if _ebay_oauth_token and time.time() < _ebay_oauth_expiry - 60:
        return _ebay_oauth_token
    EBAY_CERT_ID = os.getenv("EBAY_CERT_ID", "")
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        return ""
    try:
        creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
        r = _req.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"},
            data="grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope",
            timeout=10,
        )
        data = r.json()
        _ebay_oauth_token  = data.get("access_token", "")
        _ebay_oauth_expiry = time.time() + int(data.get("expires_in", 7200))
        return _ebay_oauth_token
    except Exception as err:
        print(f"   eBay OAuth error: {err}")
        return ""


def fetch_ebay_prices(title: str) -> dict:
    """
    Fetch both sold and active eBay listings for a title.
    Returns dict with sold_used, sold_new, active_used, active_new price lists
    and a formatted summary string for the Gemini prompt.
    """
    # Sold listings (completed)
    sold_all = _ebay_find(
        "findCompletedItems", title,
        {"itemFilter(0).name": "SoldItemsOnly", "itemFilter(0).value": "true"}
    )
    # Active BIN listings
    active_all = _ebay_find(
        "findItemsAdvanced", title,
        {"itemFilter(0).name": "ListingType", "itemFilter(0).value": "FixedPrice"}
    )

    def split_cond(items):
        used = [i["price"] for i in items if "used" in i.get("condition","").lower() or "refurb" in i.get("condition","").lower()]
        new  = [i["price"] for i in items if "new" in i.get("condition","").lower()]
        other = [i["price"] for i in items if i not in used and i not in new]
        # If no condition split, put all in used bucket
        if not used and not new:
            used = [i["price"] for i in items]
        return used, new

    sold_used_prices, sold_new_prices   = split_cond(sold_all)
    active_used_prices, active_new_prices = split_cond(active_all)

    def fmt(prices, label):
        if not prices:
            return f"{label}: no data"
        lo, hi, avg = min(prices), max(prices), sum(prices)/len(prices)
        return f"{label}: low ${lo:.2f}, high ${hi:.2f}, avg ${avg:.2f} ({len(prices)} listings)"

    summary_lines = []
    if sold_all or active_all:
        summary_lines.append("=== REAL eBay MARKET DATA (use these for pricing) ===")
        summary_lines.append(fmt(sold_used_prices,   "Sold USED"))
        summary_lines.append(fmt(sold_new_prices,    "Sold NEW"))
        summary_lines.append(fmt(active_used_prices, "Active BIN USED"))
        summary_lines.append(fmt(active_new_prices,  "Active BIN NEW"))
        # Sample titles so Gemini can see what sold
        if sold_all:
            summary_lines.append("Recent sold examples: " + " | ".join(
                f'"{i["title"][:50]}" ${i["price"]:.2f}' for i in sold_all[:3]
            ))
        if active_all:
            summary_lines.append("Active listing examples: " + " | ".join(
                f'"{i["title"][:50]}" ${i["price"]:.2f}' for i in active_all[:3]
            ))
        summary_lines.append("=====================================================")

    return {
        "sold_used":    sold_used_prices,
        "sold_new":     sold_new_prices,
        "active_used":  active_used_prices,
        "active_new":   active_new_prices,
        "summary":      "\n".join(summary_lines) if summary_lines else "",
        "has_data":     bool(sold_all or active_all),
    }

# ------------------------------------------------------------------ #
#  GEMINI PROMPT
# ------------------------------------------------------------------ #

def make_prompt(photo_count: int, condition: str = "used", ebay_data: dict = None,
                 id_title: str = "", allow_search: bool = True) -> str:
    ebay_section = ""
    if ebay_data and ebay_data.get("has_data"):
        if allow_search:
            ebay_section = f"""
{ebay_data["summary"]}

Use the eBay market data above as your PRIMARY source for pricing.
Also use Google Search to verify and supplement with additional sold listings.
Prioritize sold listings over active listings for pricing accuracy.
"""
        else:
            # API-first mode: eBay's API already returned real sold/active data, so
            # skip the search-and-verify pass entirely — just use what's given.
            ebay_section = f"""
{ebay_data["summary"]}

Use ONLY the eBay market data above for pricing — it is real sold/active data from
eBay's own API. Do not search the web to verify it. Just use these numbers directly.
Prioritize sold listings over active listings for pricing accuracy.
"""
    else:
        ebay_section = """
Use your Google Search tool to find current eBay pricing for this item. Search for:
1. eBay SOLD listings (site:ebay.com "sold" keyword) — most important for accurate pricing
2. eBay active Buy It Now listings
3. Amazon pricing
4. Any other marketplace (Reverb, OfferUp, Facebook Marketplace, etc.)
Search for both USED and NEW condition prices separately and report all prices found.
"""

    id_section = f"""
IDENTIFIED INFO: "{id_title}"

CRITICAL FIRST STEP: If the title contains a part number, search for it NOW before doing anything else.
- Search Google: "[brand] [part number] site:ebay.com" to find the real item name
- "Caterpillar 17C0033 Empty Bag" means part number is 17C0033 — search "CAT 17C0033" to find it is actually a specific seal, gasket, or component
- NEVER use words like "Empty Bag", "Packaging", "Label", "Sticker", or "Memorabilia" in the title
- The part number reveals the real item — always research it
""" if id_title else ""

    return f"""You are an expert industrial parts eBay pricing specialist analyzing {photo_count} photo(s).
{id_section}
TASK 1 — RESEARCH THE PART NUMBER:
- The pre-identified title is your starting point: "{id_title}"
- If it contains ANY part number or model number, search for it IMMEDIATELY
- Search: "[brand] [part number]" on Google and eBay to find the exact item name
- Example: "CAT 17C0033" → search → find it is a "Caterpillar 17C0033 Seal Kit" or similar
- The part number is more important than any visual description
- Replace vague titles like "Empty Bag" or "Industrial Part" with the real item name found via search
- Never keep a generic title if a part number exists that you can research

TASK 2 — PRICING RESEARCH:
{ebay_section}
The employee marked this item as: {condition}

TASK 3 — SELECT THE CORRECT EBAY CATEGORY:
- Think carefully about what this item actually is before selecting a category
- A trailer hitch goes in eBay Motors > Towing, NOT Sporting Goods
- An industrial valve goes in Business & Industrial, NOT Home & Garden
- Use your knowledge of the item type to pick the most specific accurate category
- Search eBay to verify the category if unsure

Return ONLY a raw JSON object, no markdown, no backticks:

{{
  "title": "Brand PartNumber ItemType KeySpec — keyword-rich eBay title under 80 chars",
  "ebay_category": "full eBay category path e.g. eBay Motors > Parts & Accessories > ...",
  "ebay_category_id": numeric eBay category ID as number,
  "weight_oz": estimated weight in ounces as number,
  "weight_lb": estimated weight in pounds as number,
  "price_used_low": lowest used price found as number — 0 if none,
  "price_used_high": highest used price found as number — 0 if none,
  "price_used": recommended used listing price as number — 0 if none,
  "price_new_low": lowest new price found as number — 0 if none,
  "price_new_high": highest new price found as number — 0 if none,
  "price_new": recommended new listing price as number — 0 if none
}}

ABSOLUTE RULES:
- NEVER return Unknown Item
- Keep all part numbers from the pre-identified title
- Category must match the actual item type — double check before returning"""

# ------------------------------------------------------------------ #
#  PROCESS A GROUP
# ------------------------------------------------------------------ #

def truncate_title(t: str, limit: int = 80) -> str:
    t = t.title()
    if len(t) <= limit:
        return t
    return t[:limit].rsplit(" ", 1)[0].rstrip(",.;:-")


class PartIdentification(BaseModel):
    raw_text_read: str = Field(description="Strict transcription of ALL letters, numbers, codes visible on the part. If none visible, write NONE.")
    verified_brand: str = Field(description="Brand ONLY if explicitly written in raw_text_read. Otherwise write UNBRANDED.")
    verified_part_number: str = Field(description="Exact part number ONLY if found in raw_text_read. Otherwise write UNKNOWN.")
    physical_description: str = Field(description="Physical description: material, shape, size, application.")
    generated_title: str = Field(description="Final eBay title, maximum 80 characters. Must end at a complete word — never cut off mid-word. Priority: brand + part number + item type + key specs. Drop least important words to stay under 80 chars cleanly.")


def process_group(group: dict):
    group_id  = group["id"]
    condition = group.get("condition", "used")
    quantity  = group.get("quantity", 1)
    category_mode = group.get("category_mode", "industrial")
    if category_mode not in ("industrial", "motors"):
        category_mode = "industrial"
    pricing_mode = group.get("pricing_mode", "always_search")
    if pricing_mode not in ("always_search", "api_first"):
        pricing_mode = "always_search"
    business_id = group.get("business_id")
    if not business_id:
        # Without this the listing insert lands with business_id=NULL, which makes
        # it permanently invisible to the web app's /api/listings query (it filters
        # .eq("business_id", business_id) — a null row can never match). Confirmed
        # this actually happened: two real, fully-processed scans (RÖHM live center,
        # Spicer U-joint) sat in the listings table forever with business_id=NULL
        # because nothing ever set it explicitly and whatever DB-side default/trigger
        # was supposed to fill it in didn't. Log loudly rather than silently proceed.
        print(f"   ⚠️  group {group_id} has no business_id — the listing this creates "
              f"will not be visible in the web app until this is fixed.")

    print(f"\n📦 Processing group {group_id} — condition: {condition}, qty: {quantity}, category_mode: {category_mode}")

    supabase.table("listing_groups").update(
        {"status": "processing"}
    ).eq("id", group_id).execute()

    photos_result = (
        supabase.table("group_photos")
        .select("*")
        .eq("group_id", group_id)
        .order("uploaded_at")
        .execute()
    )

    if not photos_result.data:
        print(f"   ⚠️  No photos found for group {group_id}")
        supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
        return

    photo_records = photos_result.data
    print(f"   📸 Found {len(photo_records)} photos")

    image_parts  = []
    primary_name = None
    scanned_at   = datetime.now().isoformat()

    for i, record in enumerate(photo_records):
        old_name = record["photo_id"]
        try:
            raw_bytes = supabase.storage.from_("part-photos").download(old_name)
            if i == 0:
                dt, scanned_at = get_exif_date(raw_bytes)
                new_name = build_new_filename(dt, old_name)
            else:
                dt       = datetime.now()
                new_name = build_new_filename(dt, old_name)

            jpeg_bytes = to_jpeg_bytes(raw_bytes)
            renamed    = rename_in_supabase(jpeg_bytes, old_name, new_name)
            final_name = new_name if renamed else old_name

            supabase.table("group_photos").update(
                {"photo_id": final_name}
            ).eq("id", record["id"]).execute()

            if i == 0:
                primary_name = final_name

            image_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )

            # Mark both old and new names as seen
            mark_seen(old_name)
            mark_seen(final_name)

            time.sleep(1)

        except Exception as _err:
            print(f"   ⚠️  Error processing photo {old_name}: {_err}")

    if not image_parts:
        print(f"   ⚠️  No images could be processed")
        supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
        return

    # ---- STEP 1: Structured ID pass using Pydantic schema ----
    print(f"   \U0001f50d Step 1: Identifying item from photos...")
    title_for_ebay = ""
    try:
        id_prompt = """Analyze this photo of an industrial part for eBay resale.

CRITICAL RULES:
1. READ FIRST: Transcribe all visible text, numbers, and codes exactly as they appear. Look closely at stamped metal, worn labels, cast markings.
2. NO GUESSING: Never assume a manufacturer based on color, shape, or style. If it is not written on the part, it is UNBRANDED.
   EXCEPTION: If you can clearly read a brand name on a label, sticker, or packaging, that IS the brand.
3. PACKAGING vs PART: If the item appears to be in a bag, box, or has a label — identify the PART INSIDE, not the packaging or label itself.
   - A yellow CAT bag with part number 17C0033 = a Caterpillar part #17C0033, not a "sticker"
   - A box with "Donaldson P182050" = a Donaldson filter, not a "box"
   - Always identify what the part IS, not what it comes in
3. INTERPRET CORRECTLY: Common stampings on industrial parts have specific meanings:
   - "CAP XX TONS" means capacity is XX tons e.g. "CAP 10 TONS" = 10 Ton Capacity, NOT a brand called CAP and NOT 10,500 lbs
   - "WLL XX" = working load limit, NOT a brand
   - "SWL XX" = safe working load, NOT a brand
   - "MAX XX LBS" = maximum load, NOT a brand
   - Numbers alone (e.g. "15000") = weight/load rating in lbs
   Include these as specs in the title, not as brand names.
4. CHAIN OF THOUGHT: Fill raw_text_read first, then verified_brand, then verified_part_number, then physical_description, then generated_title."""

        id_model = "models/gemini-2.5-pro"
        id_resp = None
        for _attempt in range(3):
            try:
                id_resp = call_gemini_with_timeout(lambda: client.models.generate_content(
                    model=id_model,
                    contents=[*image_parts, id_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction="You are an expert industrial parts identifier specializing in reading part numbers, model numbers, and brand names from photos. Your PRIMARY job is to find any alphanumeric codes on the item and transcribe them exactly. Part numbers are the most valuable piece of information — they unlock everything else. Even partial numbers are valuable. Read every character carefully.",
                        response_mime_type="application/json",
                        response_schema=PartIdentification,
                    )
                ))
                break
            except Exception as _e:
                err = str(_e)
                if "did not respond within" in err:
                    print(f"   \u23f3 Gemini stalled, retrying...")
                elif "503" in err or "UNAVAILABLE" in err or "429" in err:
                    print(f"   \u23f3 Gemini Pro busy, retrying in 15s...")
                    time.sleep(15)
                elif "404" in err or "deprecated" in err.lower() or "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                    id_model = model
                    print(f"   \u26a0\ufe0f  Pro unavailable, falling back to {model}")
                else:
                    raise
        if id_resp is None:
            raise Exception("Gemini unavailable after 3 retries")
        parsed_data    = json.loads(id_resp.text)
        title_for_ebay = parsed_data.get("generated_title", "").strip()
        text_found     = parsed_data.get("raw_text_read", "").strip()
        print(f"   \U0001f4dd Text found:   {text_found[:100]}")
        print(f"   \U0001f3f7\ufe0f  Brand:        {parsed_data.get('verified_brand')}")
        print(f"   \U0001f522 Part number:  {parsed_data.get('verified_part_number')}")
        print(f"   \u2705 Title:        {title_for_ebay}")
    except Exception as _err:
        print(f"   \u26a0\ufe0f  ID pass failed: {_err}")
    if not title_for_ebay:
        title_for_ebay = ""
        print(f"   ⚠️  ID pass failed: {_err}")

    # ---- STEP 2: Fetch real eBay prices (sold + active) ----
    ebay_data = {}
    if title_for_ebay != "Unknown Item":
        print(f"   📦 Fetching eBay sold + active listings via API...")
        ebay_data = fetch_ebay_prices(title_for_ebay)
        if ebay_data.get("has_data"):
            sc = len(ebay_data.get("sold_used",[])) + len(ebay_data.get("sold_new",[]))
            ac = len(ebay_data.get("active_used",[])) + len(ebay_data.get("active_new",[]))
            print(f"   ✅ eBay API: {sc} sold, {ac} active listings found")
        else:
            print(f"   ⚠️  eBay API unavailable — Gemini will search eBay + web directly")

    # ---- STEP 3: Full Gemini pass with real eBay data injected ----
    ebay_has_data = bool(ebay_data.get("has_data"))
    # always_search (default, unchanged behavior): always cross-check via Google
    # Search regardless of whether the eBay API succeeded — catches Amazon/Reverb/
    # Facebook Marketplace comps a structured eBay-only query would miss.
    # api_first (new, opt-in toggle): if the eBay API already returned real
    # sold+active data, trust it and skip the search-and-verify pass entirely —
    # this is the slow step, so skipping it when we already have good data is the
    # whole point of this mode.
    use_search = True if pricing_mode == "always_search" else (not ebay_has_data)
    prompt = make_prompt(len(image_parts), condition, ebay_data, id_title=title_for_ebay,
                          allow_search=use_search)
    print(f"   🤖 Step 3: Gemini pricing pass (mode: {pricing_mode}, "
          f"web search: {'on' if use_search else 'skipped — trusting eBay API data'})...")

    try:
        cfg_kwargs = dict(
            temperature=0.1,
            system_instruction="You are an expert industrial parts resale pricing specialist. You identify parts precisely from photos, search eBay for real sold prices, and return accurate structured data. Never guess manufacturer names — only state brands you can read on the part."
        )
        if use_search:
            cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        response = None
        for _attempt in range(3):
            try:
                response = call_gemini_with_timeout(lambda: client.models.generate_content(
                    model=model,
                    contents=[*image_parts, prompt],
                    config=cfg
                ), timeout_seconds=150)  # longer -- this pass does real web/eBay research via Google Search grounding
                break
            except Exception as _e:
                if "did not respond within" in str(_e):
                    print(f"   ⏳ Gemini stalled, retrying...")
                elif "503" in str(_e) or "UNAVAILABLE" in str(_e):
                    print(f"   ⏳ Gemini busy, retrying in 10s (attempt {_attempt+1}/3)...")
                    time.sleep(10)
                else:
                    raise
        if response is None:
            raise Exception("Gemini unavailable after 3 retries")

        # response.text can be None when Google Search tool is used
        def extract_text(resp):
            if resp is None: return ""
            try:
                if resp.text: return resp.text
            except Exception: pass
            try:
                for cand in (resp.candidates or []):
                    for part in (getattr(cand.content, "parts", None) or []):
                        t = getattr(part, "text", None)
                        if t: return t
            except Exception: pass
            return ""
        raw = extract_text(response)
        raw = (raw or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw).strip()

        data             = json.loads(raw)
        title            = str(data.get("title", "Unknown Item")).strip()[:80]
        ebay_category    = str(data.get("ebay_category", "")).strip()
        ebay_category_id = str(parse_int(data.get("ebay_category_id", 0)))
        weight_oz        = parse_num(data.get("weight_oz", 0))
        weight_lb        = parse_num(data.get("weight_lb", 0))
        price_used_low   = parse_num(data.get("price_used_low", 0))
        price_used_high  = parse_num(data.get("price_used_high", 0))
        price_used       = parse_num(data.get("price_used", 0))
        price_new_low    = parse_num(data.get("price_new_low", 0))
        price_new_high   = parse_num(data.get("price_new_high", 0))
        price_new        = parse_num(data.get("price_new", 0))

        # ---- eBay Category Suggestion API ----
        # Single lane, decided by the intake toggle (category_mode, read off the
        # group above). No cross-tree fallback: an item marked Auto Parts stays in
        # tree 100 and lands on the Motors catch-all if nothing matches, rather than
        # silently drifting into Business & Industrial. The old version queried both
        # trees but appended tree 0 first and took the first acceptable hit, which
        # made tree 100 unreachable in practice for anything tree 0 could answer.
        MOTORS_FALLBACK_ID = os.environ.get("EBAY_DEFAULT_MOTORS_CATEGORY_ID", "").strip()
        if category_mode == "motors":
            _tree_id = "100"
            ebay_category_id = MOTORS_FALLBACK_ID or "26261"
            ebay_category = "Other (eBay Motors)" if MOTORS_FALLBACK_ID else "Other Business & Industrial"
            if not MOTORS_FALLBACK_ID:
                print("   ⚠️  EBAY_DEFAULT_MOTORS_CATEGORY_ID env var not set on mantle-scanner "
                      "— Auto Parts misses will land on 26261 instead of the Motors catch-all.")
        else:
            _tree_id = "0"
            ebay_category_id = "26261"  # Other Business & Industrial
            ebay_category = "Other Business & Industrial"

        if title and title != "Unknown Item" and EBAY_APP_ID:
            try:
                import requests as _req
                token = get_ebay_token()
                cat_resp = _req.get(
                    f"https://api.ebay.com/commerce/taxonomy/v1/category_tree/{_tree_id}/get_category_suggestions",
                    params={"q": title},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    timeout=5
                )
                if cat_resp.status_code != 200:
                    print(f"   ⚠️  Category tree {_tree_id} lookup returned "
                          f"{cat_resp.status_code}: {cat_resp.text[:200]}")
                else:
                    chosen = None
                    for s in cat_resp.json().get("categorySuggestions", []):
                        if _tree_id == "100":
                            # Tree 100 IS eBay Motors by definition — no name check needed.
                            chosen = s
                            break
                        ancestors = s.get("categoryTreeNodeAncestors", []) or []
                        # NOTE: eBay returns ancestors nearest-parent-first, so the ROOT
                        # (top-level) category is the LAST element, not the first. The
                        # previous version read ancestors[0] here, which is the item's
                        # immediate parent — comparing that against "Business & Industrial"
                        # almost never matched, silently rejecting most valid B&I results
                        # and pushing them to the 26261 fallback below.
                        top_level_name = (ancestors[-1].get("categoryName", "") if ancestors
                                           else s.get("category", {}).get("categoryName", "")).lower()
                        if top_level_name == "business & industrial":
                            chosen = s
                            break
                    if chosen:
                        best = chosen["category"]
                        ebay_category_id = str(best.get("categoryId", ebay_category_id))
                        ebay_category    = best.get("categoryName", ebay_category)
                        print(f"   📂 eBay category: {ebay_category} (ID: {ebay_category_id}, "
                              f"tree {_tree_id}, mode {category_mode})")
                    else:
                        print(f"   ⚠️  No match in tree {_tree_id} (mode {category_mode}) — "
                              f"locked to catch-all category {ebay_category_id}")
            except Exception as _ce:
                print(f"   ⚠️  Category lookup failed: {_ce}")


        if condition == "used":
            active_price = price_used if price_used > 0 else price_new
            active_low   = price_used_low if price_used_low > 0 else price_new_low
            active_high  = price_used_high if price_used_high > 0 else price_new_high
            price_note   = "new" if price_used == 0 and price_new > 0 else ""
        else:
            active_price = price_new if price_new > 0 else price_used
            active_low   = price_new_low if price_new_low > 0 else price_used_low
            active_high  = price_new_high if price_new_high > 0 else price_used_high
            price_note   = "used" if price_new == 0 and price_used > 0 else ""

    except Exception as _gemini_err:
        print(f"   ⚠️  Gemini error: {_gemini_err}")
        title, ebay_category, price_note = title_for_ebay or "Industrial Part", "", ""
        ebay_category_id = "0"
        weight_oz = weight_lb = 0.00
        price_used = price_used_low = price_used_high = 0.00
        price_new  = price_new_low  = price_new_high  = 0.00
        active_price = active_low = active_high = 0.00

    _listing_payload = {
        "title":            title,
        "ebay_category":    ebay_category,
        "ebay_category_id": ebay_category_id,
        "category_mode":    category_mode,
        "pricing_mode":     pricing_mode,
        "pricing_used_search": use_search,
        "business_id":      business_id,
        "weight_oz":        weight_oz,
        "weight_lb":        weight_lb,
        "price_low":        active_low,
        "price_high":       active_high,
        "price":            active_price,
        "price_note":       price_note,
        "price_used":       price_used,
        "price_new":        price_new,
        "photo_id":         primary_name,
        "quantity":         quantity,
        "condition":        condition,
        "status":           "scanned",
        "created_at":       scanned_at,
    }
    try:
        _ins_res = supabase.table("listings").insert(_listing_payload).execute()
        if not getattr(_ins_res, "data", None):
            raise Exception(f"insert returned no data: {_ins_res}")
    except Exception as _ins_err:
        # Most likely cause of a silent-looking failure right after a migration:
        # PostgREST's schema cache hasn't picked up a newly added column yet
        # (category_mode), so it rejects the whole insert. Retry once without
        # it rather than losing the scan entirely — this listing just won't
        # carry its mode until the cache catches up or the next rescan.
        print(f"   ⚠️  listings insert failed ({_ins_err}) — retrying without category_mode "
              f"in case the column isn't in PostgREST's schema cache yet")
        _listing_payload.pop("category_mode", None)
        _listing_payload.pop("pricing_mode", None)
        _listing_payload.pop("pricing_used_search", None)
        _ins_res2 = supabase.table("listings").insert(_listing_payload).execute()
        if not getattr(_ins_res2, "data", None):
            print(f"   ❌ listings insert failed again — this scan's listing was NOT created: {_ins_res2}")
            supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
            return

    supabase.table("listing_groups").update({"status": "done"}).eq("id", group_id).execute()

    print(f"   ✅ {title}")
    print(f"   SKU      : {primary_name}")
    print(f"   Category : {ebay_category} (ID: {ebay_category_id})")
    print(f"   Used     : ${price_used:.2f} / New: ${price_new:.2f}")
    print(f"   Active   : ${active_price:.2f}{' (' + price_note + ')' if price_note else ''}")
    print(f"   Quantity : {quantity}")

# ------------------------------------------------------------------ #
#  LEGACY SINGLE-PHOTO WATCHER
# ------------------------------------------------------------------ #

def process_legacy_photo(file_info):
    old_name = file_info['name']
    print(f"📸 Legacy scan: {old_name}")

    try:
        raw_bytes      = supabase.storage.from_("part-photos").download(old_name)
        dt, scanned_at = get_exif_date(raw_bytes)
        new_name       = build_new_filename(dt, old_name)
        jpeg_bytes     = to_jpeg_bytes(raw_bytes)
        renamed        = rename_in_supabase(jpeg_bytes, old_name, new_name)
        photo_id       = new_name if renamed else old_name

        # Mark both as seen immediately
        mark_seen(old_name)
        mark_seen(photo_id)

        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        condition  = "used"  # legacy single-photo path has no condition input -- always "used", same
                              # default make_prompt itself uses. Previously passed directly into
                              # make_prompt() without ever being saved to a real variable, so the later
                              # listings insert (which references `condition` by name) raised a bare
                              # NameError on every legacy scan -- hidden until now behind the billing
                              # 403s, which killed every request before this code ever ran.
        prompt     = make_prompt(1, condition)

        response = call_gemini_with_timeout(lambda: client.models.generate_content(
            model=model,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        ), timeout_seconds=150)

        # response.text can be None when Google Search tool is used
        def extract_text(resp):
            if resp is None: return ""
            try:
                if resp.text: return resp.text
            except Exception: pass
            try:
                for cand in (resp.candidates or []):
                    for part in (getattr(cand.content, "parts", None) or []):
                        t = getattr(part, "text", None)
                        if t: return t
            except Exception: pass
            return ""
        raw = extract_text(response)
        raw = (raw or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw).strip()

        data             = json.loads(raw)
        title            = str(data.get("title", "Unknown Item")).strip()[:80]
        ebay_category    = str(data.get("ebay_category", "")).strip()
        ebay_category_id = str(parse_int(data.get("ebay_category_id", 0)))
        weight_oz        = parse_num(data.get("weight_oz", 0))
        weight_lb        = parse_num(data.get("weight_lb", 0))
        price_used       = parse_num(data.get("price_used", 0))
        price_used_low   = parse_num(data.get("price_used_low", 0))
        price_used_high  = parse_num(data.get("price_used_high", 0))
        price_new        = parse_num(data.get("price_new", 0))
        price_new_low    = parse_num(data.get("price_new_low", 0))
        price_new_high   = parse_num(data.get("price_new_high", 0))

        active_price = price_used if price_used > 0 else price_new
        active_low   = price_used_low if price_used_low > 0 else price_new_low
        active_high  = price_used_high if price_used_high > 0 else price_new_high
        price_note   = "new" if price_used == 0 and price_new > 0 else ""

        supabase.table("listings").insert({
            "title":            title,
            "ebay_category":    ebay_category,
            "ebay_category_id": ebay_category_id,
            "weight_oz":        weight_oz,
            "weight_lb":        weight_lb,
            "price_low":        active_low,
            "price_high":       active_high,
            "price":            active_price,
            "price_note":       price_note,
            "price_used":       price_used,
            "price_new":        price_new,
            "photo_id":         photo_id,
            "condition":        condition,
            "status":           "scanned",
            "created_at":       scanned_at,
        }).execute()

        print(f"   ✅ {title} — used: ${price_used:.2f} / new: ${price_new:.2f}")

    except Exception as _err:
        print(f"   ⚠️  Error: {_err}")

# ------------------------------------------------------------------ #
#  WATCHER LOOP
# ------------------------------------------------------------------ #

print("🕵️  Lister AI ACTIVE... Watching for groups and photos.")
print("📋 Loading seen files from Supabase...")

seen_files = load_seen()
print(f"📋 {len(seen_files)} files already seen.")

# Seed seen_files with everything currently in storage on first run
if len(seen_files) == 0:
    try:
        existing = {f['name'] for f in supabase.storage.from_("part-photos").list()}
        for fname in existing:
            mark_seen(fname)
        seen_files = existing
        print(f"📋 Seeded {len(seen_files)} existing files — watching for new ones only.")
    except Exception as _err:
        print(f"⚠️  Could not seed existing files: {_err}")

while True:
    try:
        # 1. Reload seen files from Supabase every loop
        seen_files = load_seen()

        # 2. Check for pending groups
        pending = (
            supabase.table("listing_groups")
            .select("*")
            .eq("status", "pending")
            .execute()
        )
        for group in (pending.data or []):
            process_group(group)

        # 3. Check for legacy single photos. Interleave a pending-groups check
        # every few legacy photos -- a large legacy backlog (this can run for
        # many minutes/hours straight) would otherwise starve real, current
        # group scans entirely, since nothing re-checked "pending" until this
        # whole for-loop finished. Confirmed this actually happened: a full
        # deploy ran 4+ minutes processing legacy photos non-stop without
        # ever once touching a "Processing group" call, while a real pending
        # group sat waiting the entire time.
        current = supabase.storage.from_("part-photos").list()
        for _legacy_idx, f in enumerate(current):
            if f['name'] not in seen_files:
                # Check if this photo belongs to a group
                group_check = (
                    supabase.table("group_photos")
                    .select("id")
                    .eq("photo_id", f['name'])
                    .execute()
                )
                if not group_check.data:
                    process_legacy_photo(f)
                else:
                    # Part of a group — just mark as seen
                    mark_seen(f['name'])

                # Interleaved check, every 3 legacy photos rather than every
                # single one (keeps the extra query overhead low while still
                # keeping real group scans from waiting an unbounded amount
                # of time behind a large legacy backlog).
                if (_legacy_idx + 1) % 3 == 0:
                    interleaved_pending = (
                        supabase.table("listing_groups")
                        .select("*")
                        .eq("status", "pending")
                        .execute()
                    )
                    for group in (interleaved_pending.data or []):
                        process_group(group)

    except Exception as _err:
        print(f"⚠️  Connection hiccup: {_err}")

    time.sleep(5)
