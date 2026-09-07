import os
from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_SERVICE_KEY not found."
    )

supabase = create_client(url, key)

response = (
    supabase
    .table("customer_risk_scoring")
    .select("*")
    .limit(5)
    .execute()
)

print(response.data)