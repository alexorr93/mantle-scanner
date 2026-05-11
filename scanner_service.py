import os
import re
import time
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
    Call eBay Finding API and return list of item dicts with price + title + url.
    operation: findCompletedItems | findItemsAdvanced (Browse API)
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

def make_prompt(photo_count: int, condition: str = "used", ebay_data: dict = None, id_title: str = "") -> str:
    if ebay_data and ebay_data.get("has_data"):
        market_section = f"""EBAY MARKET DATA (PRIMARY SOURCE):
{ebay_data["summary"]}

Use these prices as your foundation. Supplement with Google Search for any missing sold data."""
    else:
        market_section = """MARKET RESEARCH — RUN GOOGLE SEARCHES IN THIS ORDER:
1. "[item name] sold ebay" — find completed/sold listings
2. "[item name] site:amazon.com" — find current retail price
3. "[item name] ebay" — find active Buy It Now only as last resort"""

    id_section = f"""PRE-IDENTIFIED TITLE: "{id_title}"
Verify correctness via research. Improve if research reveals the actual product name.
NEVER use: Empty, Bag, Industrial Part, Unknown Item.""" if id_title else ""

    return f"""You are an elite secondary market pricing actuary. Your objective is the TRUE liquid cash value — what a buyer will actually pay on eBay today — not asking prices.

Analyzing {photo_count} photo(s) for eBay resale.
{id_section}

Condition marked: {condition.upper()}

{market_section}

=== PHASE 1: DEEP IDENTIFICATION ===
Before pricing, lock in the EXACT item. Identify:
- Brand & core product name
- Variant anchors: size/dimensions, year of release, edition type (Open vs Limited), model number, material
- If the photo does not provide enough detail to confirm the exact variant, flag is_ambiguous=true

=== PHASE 2: CATEGORY TRIAGE ===
Classify into ONE economic model:
- MODEL A (Standard/Electronics/Tools/Industrial): Subject to depreciation. Resale value CANNOT exceed current retail.
- MODEL B (Collectibles/Hype/Art/Vintage/Trading Cards/Sneakers/Designer): Subject to scarcity economics. Resale CAN and OFTEN DOES exceed original retail. Examples: KAWS, Nike limited editions, Pokémon cards, vintage watches, sports cards, Funko grails.

=== PHASE 3: FORKED WATERFALL PROTOCOL ===

IF MODEL A (Standard):
TIER 1 — VERIFIED SOLD COMPS (Gold Standard)
Search eBay completed/sold listings last 90 days. Extract ALL prices, discard outliers, average the rest.
- If Tier 1 found: SET pricing_tier="SOLD_COMPS" and STOP.

TIER 2 — RETAIL REALITY CHECK
Search Amazon, Walmart, manufacturer site.
- Used resale = 55-70% of retail. New = 75-85% of retail.
- CRITICAL for MODEL A only: resale CANNOT exceed retail.
- If Tier 2 found: SET pricing_tier="RETAIL_ANCHORED" and STOP.

TIER 3 — INDUSTRIAL DEALERS
Grainger, MSC, Radwell for specialty items.
- Used = 50-60% of dealer price. New = 70-80%.
- If Tier 3 found: SET pricing_tier="DEALER_DISCOUNTED" and STOP.

IF MODEL B (Collectible/Hype):
TIER 1 — SOLD COMPS ONLY (Mandatory)
Search eBay sold, StockX, Heritage Auctions, PWCC for recent sales.
- Extract minimum 3 comparable data points. Discard fakes/condition anomalies. Average the rest.
- DO NOT use retail as a ceiling. Secondary market price IS the true value.
- If Tier 1 found: SET pricing_tier="SOLD_COMPS" and STOP.
- SKIP Tier 2 entirely — retail is irrelevant for collectibles.

TIER 4 — ACTIVE LISTINGS (Last resort, both models)
Only if Tiers 1-3 completely fail.
- Take LOWEST active listing × 0.60.
- SET pricing_tier="ACTIVE_LISTINGS".

TIER 3 — INDUSTRIAL/SPECIALTY DEALERS
For specialized items with no retail or sold comps (Grainger, MSC, Radwell, surplus sites).
- Used resale = strictly 50% to 60% of dealer listed price.
- New/sealed resale = 70% to 80% of dealer listed price.
- If Tier 3 data found: SET pricing_tier="DEALER_DISCOUNTED" and STOP HERE.

TIER 4 — ACTIVE LISTINGS (Last Resort Only)
Only use if Tiers 1-3 all fail completely.
- Active listings are ASKING prices, not reality. Sellers routinely overprice.
- Take the LOWEST active listing and multiply by 0.60 to get true value.
- SET pricing_tier="ACTIVE_LISTINGS".

TIER 5 — REASONED ESTIMATE
If zero market data found anywhere:
- Use category knowledge for a starting estimate.
- Consumer electronics used: $10-40 typical range.
- Clothing used: $5-25, new: $10-40.
- Tools used: $10-50, new: $20-80.
- SET pricing_tier="ESTIMATED". Never output $0 unless item is genuinely unsellable.

AGGREGATION RULES:
- Never anchor to a single price. Always extract multiple data points.
- Always show your math in data_sources_count.
- USED and NEW prices must be researched separately.
- You MUST always provide BOTH price_used AND price_new. Never leave either as 0.
- If you can only find one condition, estimate the other: used = new x 0.65, new = used / 0.65.

EBAY CATEGORIES:
Consumer Electronics: 293, Clothing: 11450, Tools: 631, Toys: 220, Collectibles: 1
PLCs: 115708, Sensors: 78189, Hydraulic valves: 98463, Pumps: 12576
Motors: 124660, VFDs: 115082, Circuit breakers: 66825, Contactors: 66828

=== PHASE 4: AGGREGATION RULES ===
- Never anchor to a single price. Extract minimum 3 data points.
- For MODEL B: if price spread exceeds 3x (e.g. $100 to $300+), flag requires_manual_review=true.
- USED and NEW prices must be researched separately.
- Always provide BOTH price_used AND price_new. Estimate missing one: used = new × 0.65.

OUTPUT — Return ONLY raw JSON, no markdown:

{{
  "title": "Specific eBay title under 80 chars — include brand, exact model/variant, size, year if known",
  "ebay_category": "Full category path",
  "ebay_category_id": <number>,
  "weight_oz": <number>,
  "weight_lb": <number>,
  "price_used_low": <number>,
  "price_used_high": <number>,
  "price_used": <number>,
  "price_new_low": <number>,
  "price_new_high": <number>,
  "price_new": <number>,
  "pricing_tier": "SOLD_COMPS" or "RETAIL_ANCHORED" or "DEALER_DISCOUNTED" or "ACTIVE_LISTINGS" or "ESTIMATED",
  "data_sources_count": <number>,
  "economic_model": "MODEL_A" or "MODEL_B",
  "is_ambiguous": true or false,
  "confidence_score": <number 1-100>,
  "requires_manual_review": true or false,
  "review_reason": "string or empty"
}}
"""

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


def _extract_brand_from_title(title: str) -> str | None:
    """Extract brand from title when visual identification fails."""
    if not title:
        return None
    known_brands = [
        "Nike","Adidas","Jordan","Yeezy","New Balance","Puma","Reebok","Vans","Converse",
        "KAWS","Medicom","Bearbrick","Funko","LEGO","Hot Wheels","Mattel","Hasbro",
        "Apple","Samsung","Sony","Bose","JBL","Beats","Dyson","Dell","HP","Lenovo","Asus",
        "Nintendo","PlayStation","Xbox","Atari","Sega",
        "Louis Vuitton","Gucci","Coach","Michael Kors","Kate Spade","Tory Burch",
        "Supreme","Palace","Stussy","Off-White","Fear of God","Carhartt","Patagonia","Arc'teryx",
        "Rolex","Omega","Seiko","Casio","Fossil","Timex","TAG Heuer",
        "Pokemon","Topps","Panini","Upper Deck","Leaf",
        "DeWalt","Milwaukee","Makita","Bosch","Ryobi","Craftsman","Stanley",
    ]
    title_lower = title.lower()
    for brand in known_brands:
        if brand.lower() in title_lower:
            return brand
    return None


def _fill_aspects_at_scan(title: str, brand: str, part_number: str, category_id: str) -> dict:
    """Fetch required eBay aspects for category and fill them with Gemini. Called at scan time."""
    import requests as _rq, json as _j
    aspects = {}
    try:
        # Get app-level token
        import base64 as _b64
        creds = _b64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
        tok_r = _rq.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"},
            data="grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope",
            timeout=8
        )
        app_token = tok_r.json().get("access_token", "")
        if not app_token:
            return aspects

        # Fetch required aspects for this category
        asp_r = _rq.get(
            f"https://api.ebay.com/commerce/taxonomy/v1/category_tree/0/get_item_aspects_for_category?category_id={category_id}",
            headers={"Authorization": f"Bearer {app_token}"},
            timeout=8
        )
        if not asp_r.ok:
            return aspects

        required = []
        recommended = []
        for asp in asp_r.json().get("aspects", []):
            name = asp.get("localizedAspectName", "")
            usage = asp.get("aspectConstraint", {}).get("aspectUsage", "OPTIONAL")
            mode  = asp.get("aspectConstraint", {}).get("aspectMode", "FREE_TEXT")
            vals  = [v.get("localizedValue","") for v in asp.get("aspectValues", [])[:12] if v.get("localizedValue")]
            if usage == "REQUIRED":
                required.append({"name": name, "mode": mode, "values": vals})
            elif usage == "RECOMMENDED":
                recommended.append({"name": name, "mode": mode, "values": vals})

        ask = required + recommended[:4]
        if not ask:
            return aspects

        # Build Gemini prompt
        fields = []
        for a in ask:
            if a["mode"] == "SELECTION_ONLY" and a["values"]:
                fields.append(f'  "{a["name"]}": choose best from {a["values"][:8]}')
            else:
                fields.append(f'  "{a["name"]}": short accurate value')

        prompt = f"""You are an expert eBay reseller. Fill in item specifics for this product.

Title: {title}
Brand: {brand or "Unbranded"}
Part/Model number: {part_number or "N/A"}

Rules:
- For SELECTION fields, pick the BEST matching option from the list given
- If genuinely unknown, use "Does Not Apply" for required fields
- Never use placeholder values or random options
- Keep values concise and accurate

Return ONLY raw JSON (no markdown):
{{
{chr(10).join(fields)}
}}"""

        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        raw = (resp.text or "").strip().replace("```json","").replace("```","").strip()
        filled = _j.loads(raw)

        # Smart fallbacks for common required fields
        FALLBACKS = {
            "Brand": brand or "Unbranded",
            "MPN": part_number or "Does Not Apply",
            "Model": part_number or "Does Not Apply",
            "Type": "Other",
            "Color": "Multicolor",
            "Department": "Unisex Adults",
            "Size": "One Size",
            "Size Type": "Regular",
            "US Shoe Size": "10",
            "Connectivity": "Wireless",
            "Country of Origin": "China",
            "Material": "Mixed Materials",
        }
        for a in required:
            name = a["name"]
            val = filled.get(name, "")
            if not val or val in ("Does Not Apply", "N/A", "Unknown", ""):
                if name in FALLBACKS:
                    filled[name] = FALLBACKS[name]
                elif a["mode"] == "SELECTION_ONLY" and a["values"]:
                    filled[name] = a["values"][0]
                else:
                    filled[name] = "Does Not Apply"
            aspects[name] = filled[name]

        # Also add recommended if filled
        for a in recommended[:4]:
            name = a["name"]
            if filled.get(name) and filled[name] not in ("", "Does Not Apply"):
                aspects[name] = filled[name]

        print(f"   🏷️  Aspects filled: {list(aspects.keys())}")
    except Exception as e:
        print(f"   ⚠️  Aspect fill error: {e}")
    return aspects


def process_group(group: dict):
    group_id    = group["id"]
    condition   = group.get("condition", "used")
    quantity    = group.get("quantity", 1)
    business_id = group.get("business_id")

    print(f"\n📦 Processing group {group_id} — condition: {condition}, qty: {quantity}")

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
        id_prompt = """You are an expert item identifier for eBay resale. Your job is to identify what the item is, read any text, model numbers, brand names, or part numbers visible in the photo.

STAGE 1 - TRANSCRIBE
Read every character on the item. Treat this like evidence - transcribe exactly what is written, not what makes sense:
- Stamped metal (often faint, look for raised or recessed characters)
- Printed labels and stickers (even partial ones)
- Cast markings (raised letters molded into the part)
- Hand-written tags
- QR codes or barcodes (note their presence even if unreadable)

Format: List every text element separated by ' | '. Example: 'PARKER | D1VW4CNYP | 24V DC | MADE IN USA | LOT 4521'
If absolutely nothing is readable, write: NONE

STAGE 2 - BRAND
The brand is ONLY what is explicitly printed/stamped on the part itself. NO inference allowed.

VALID brand sources:
- Logo or wordmark on the part
- Manufacturer label or sticker
- Cast company name in metal
- Original packaging if part is clearly inside it (CAT bag means Caterpillar)

INVALID - do NOT use these as brand:
- Color or paint scheme
- Shape or style resemblance
- Distributor names (Grainger, McMaster, MSC are sellers not brands)

If no brand is verifiable: UNBRANDED

STAGE 3 - PART NUMBER
The part number is the most valuable piece of data. It unlocks identification, pricing, and category.

Look for alphanumeric strings that match these patterns:
- 6-12 character codes with mixed letters/numbers (D1VW4CNYP, 1756-IF16, 6ES7-321)
- P/N, MODEL, CAT NO, ORDER NO prefixes
- Codes near barcodes
- Codes etched or engraved

Common formats by industry:
- Allen Bradley: 4-digit-XX (1756-IF16, 1769-L33ER)
- Siemens: 6ES7 followed by code
- Parker: Letter prefix + numbers (D1VW4CNYP)
- Caterpillar: Numbers + letters (17C0033, 1R-0750)
- SKF/FAG bearings: Numeric codes (6203-2RS)

If multiple codes, the PART NUMBER is usually the longest/most prominent. Other codes may be lot numbers, date codes, or stock numbers.

If no part number is verifiable: UNKNOWN

STAGE 4 - PHYSICAL DESCRIPTION
Describe what you SEE in 1 sentence: form factor, size estimate, condition, distinguishing features.

STAGE 5 - GENERATED TITLE
Build the title in this exact priority order, dropping items only when 80 chars is exceeded:

[BRAND] [PART_NUMBER] [ITEM_TYPE] [KEY_SPEC] [CONDITION_DESCRIPTOR]

Rules:
1. NEVER cut off mid-word - end at a clean word boundary
2. Item type must be SPECIFIC (Solenoid Valve not Valve)
3. Key spec is the most marketable detail: voltage, port size, capacity, range
4. NEVER use these words: Empty, Box, Bag, Packaging, Sticker, Label, Memorabilia, Lot of
5. If part is in packaging, identify the part inside (CAT 17C0033 bag means Caterpillar 17C0033 Seal Kit)
6. Title Case format

CHAIN OF THOUGHT: Fill raw_text_read first, then verified_brand, then verified_part_number, then physical_description, then generated_title."""

        id_model = "models/gemini-2.5-flash"
        id_resp = None
        for _attempt in range(3):
            try:
                id_resp = client.models.generate_content(
                    model=id_model,
                    contents=[*image_parts, id_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction="You are an expert item identifier for eBay resale. Identify what the item is, its brand, model, and any alphanumeric codes visible. For all items read text carefully. For parts and electronics, part numbers unlock pricing — transcribe them exactly.",
                        response_mime_type="application/json",
                        response_schema=PartIdentification,
                    )
                )
                break
            except Exception as _e:
                err = str(_e)
                if "503" in err or "UNAVAILABLE" in err or "429" in err:
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
        verified_brand = parsed_data.get("verified_brand", "").strip()
        verified_pn    = parsed_data.get("verified_part_number", "").strip()

        # Flash-to-Pro escalation: if flash returned a weak result (no brand AND
        # no part number), retry once with Pro for a more thorough read.
        # Default flash gives us speed; Pro fallback gives accuracy on hard reads.
        flash_was_weak = (not verified_brand) and (not verified_pn)
        if id_model == "models/gemini-2.5-flash" and flash_was_weak:
            print(f"   \U0001f504 Flash result low-confidence (no brand/PN), escalating to Pro...")
            try:
                pro_resp = client.models.generate_content(
                    model="models/gemini-2.5-pro",
                    contents=[*image_parts, id_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction="You are an expert item identifier for eBay resale. Identify what the item is, its brand, model, and any alphanumeric codes visible. For all items read text carefully. For parts and electronics, part numbers unlock pricing — transcribe them exactly.",
                        response_mime_type="application/json",
                        response_schema=PartIdentification,
                    )
                )
                pro_parsed = json.loads(pro_resp.text)
                if pro_parsed.get("verified_brand", "").strip() or pro_parsed.get("verified_part_number", "").strip():
                    parsed_data    = pro_parsed
                    title_for_ebay = parsed_data.get("generated_title", "").strip()
                    text_found     = parsed_data.get("raw_text_read", "").strip()
                    print(f"   \u2705 Pro escalation succeeded")
                else:
                    print(f"   \u2139\ufe0f  Pro returned no additional info, keeping flash result")
            except Exception as _esc_err:
                print(f"   \u26a0\ufe0f  Pro escalation failed (keeping flash result): {_esc_err}")

        print(f"   \U0001f4dd Text found:   {text_found[:100]}")
        print(f"   \U0001f3f7\ufe0f  Brand:        {parsed_data.get('verified_brand')}")
        print(f"   \U0001f522 Part number:  {parsed_data.get('verified_part_number')}")
        print(f"   \u2705 Title:        {title_for_ebay}")
    except Exception as _err:
        id_err_msg = str(_err)
        print(f"   \u26a0\ufe0f  ID pass failed: {id_err_msg}")
    if not title_for_ebay:
        title_for_ebay = ""

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

    # Initialize default values — these get overwritten by Gemini's response, but if
    # Gemini fails entirely we still need them to exist for the sold_count calculation.
    pricing_tier = ""
    data_sources_n = 0

    # ---- STEP 3: Full Gemini pass with real eBay data injected ----
    prompt = make_prompt(len(image_parts), condition, ebay_data, id_title=title_for_ebay)
    # Always use Google Search — it finds eBay sold listings, Amazon, and other
    # marketplaces regardless of whether the eBay API succeeded or failed
    use_search = True
    print(f"   🤖 Step 3: Gemini pricing pass (web search: always on)...")

    try:
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
            system_instruction="You are an expert resale pricing specialist. You identify items precisely from photos, search eBay for real sold prices, and return accurate structured data. Never guess brand names — only state brands you can read on the item."
        )
        response = None
        for _attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[*image_parts, prompt],
                    config=cfg
                )
                break
            except Exception as _e:
                if "503" in str(_e) or "UNAVAILABLE" in str(_e):
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
        pricing_tier    = str(data.get("pricing_tier", "")).strip().upper()
        data_sources_n  = parse_int(data.get("data_sources_count", 0))

        # ---- eBay Category Suggestion API ----
        if title and title != "Unknown Item" and EBAY_APP_ID:
            try:
                import requests as _req
                cat_resp = _req.get(
                    "https://api.ebay.com/commerce/taxonomy/v1/category_tree/0/get_category_suggestions",
                    params={"q": title},
                    headers={
                        "Authorization": f"Bearer {get_ebay_token()}",
                        "Content-Type": "application/json"
                    },
                    timeout=5
                )
                if cat_resp.status_code == 200:
                    suggestions = cat_resp.json().get("categorySuggestions", [])
                    if suggestions:
                        best = suggestions[0]["category"]
                        ebay_category_id = str(best.get("categoryId", ebay_category_id))
                        ebay_category    = best.get("categoryName", ebay_category)
                        print(f"   📂 eBay category: {ebay_category} (ID: {ebay_category_id})")
            except Exception as _ce:
                print(f"   ⚠️  Category lookup failed: {_ce}")

        # ── Fill eBay item specifics at scan time ──────────────────────
        ebay_item_specifics = {}
        if ebay_category_id and ebay_category_id not in ("0", ""):
            try:
                ebay_item_specifics = _fill_aspects_at_scan(
                    title=title,
                    brand=verified_brand if verified_brand and verified_brand != "UNBRANDED" else "",
                    part_number=verified_pn if verified_pn and verified_pn != "UNKNOWN" else "",
                    category_id=ebay_category_id
                )
            except Exception as _asp_e:
                print(f"   ⚠️  Scan-time aspect fill failed: {_asp_e}")

        # Derive missing prices — if only one is set, estimate the other
        if price_new > 0 and price_used == 0:
            price_used = round(price_new * 0.65, 2)
            price_used_low = round(price_new * 0.55, 2)
            price_used_high = round(price_new * 0.75, 2)
            print(f"   📊 Derived used price from new: ${price_used}")
        elif price_used > 0 and price_new == 0:
            price_new = round(price_used / 0.65, 2)
            price_new_low = round(price_used / 0.75, 2)
            price_new_high = round(price_used / 0.55, 2)
            print(f"   📊 Derived new price from used: ${price_new}")

        # Sanity check: used price should never exceed new price
        # If Gemini returns inverted values, swap them
        if price_used > 0 and price_new > 0 and price_used > price_new:
            print(f"   ⚠️  Price sanity: used (${price_used}) > new (${price_new}), swapping")
            price_used, price_new = price_new, price_used
            price_used_low, price_new_low = price_new_low, price_used_low
            price_used_high, price_new_high = price_new_high, price_used_high

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

    # Sold count: prefer eBay API data, fall back to Gemini's web-search confirmation
    api_sc = (len(ebay_data.get("sold_used", []) or []) + len(ebay_data.get("sold_new", []) or [])) if isinstance(ebay_data, dict) else 0
    if api_sc > 0:
        sold_count = api_sc
    elif pricing_tier in ("SOLD_COMPS", "ACTIVE_LISTINGS", "DEALER_DISCOUNTED"):
        sold_count = max(data_sources_n, 1)
    else:
        sold_count = 0
    supabase.table("listings").insert({
        "business_id":      business_id,
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
        "photo_id":         primary_name,
        "quantity":         quantity,
        "condition":        condition,
        "status":           "scanned",
        "sold_count":       sold_count,
        "created_at":       scanned_at,
        "brand":            (verified_brand if verified_brand and verified_brand not in ("UNBRANDED", "UNKNOWN") else None) or _extract_brand_from_title(title),
        "mpn":              verified_pn if verified_pn and verified_pn != "UNKNOWN" else None,
        "model":            verified_pn if verified_pn and verified_pn != "UNKNOWN" else None,
        "ebay_item_specifics": ebay_item_specifics if ebay_item_specifics else None,
    }).execute()

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
    # Skip system files (underscore-prefixed) and intake photos (uploaded for inventory tracking, not scanning)
    base = old_name.split("/")[-1] if "/" in old_name else old_name
    if base.startswith("_") or base.startswith("intake_"):
        return
    # Skip internal/system paths (anything starting with underscore)
    if old_name.startswith("_") or "/" in old_name and old_name.split("/")[-1].startswith("_"):
        return
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
        prompt     = make_prompt(1, "used")

        response = client.models.generate_content(
            model=model,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

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

        # Sold count: prefer eBay API data, fall back to Gemini's web-search confirmation
        api_sc = (len(ebay_data.get("sold_used", []) or []) + len(ebay_data.get("sold_new", []) or [])) if isinstance(ebay_data, dict) else 0
        if api_sc > 0:
            sold_count = api_sc
        elif pricing_tier in ("SOLD_COMPS", "ACTIVE_LISTINGS", "DEALER_DISCOUNTED"):
            sold_count = max(data_sources_n, 1)
        else:
            sold_count = 0
        supabase.table("listings").insert({
            "business_id":      business_id,
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
            "sold_count":       sold_count,
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

        # 3. Check for legacy single photos
        current = supabase.storage.from_("part-photos").list()
        for f in current:
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

    except Exception as _err:
        print(f"⚠️  Connection hiccup: {_err}")

    time.sleep(5)
