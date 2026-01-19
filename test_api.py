import requests

BASE_URL = "http://localhost:3000"

def test_api():
    print("Testing TraceAI API...")
    
    # 1. Test /api/cases (empty initially)
    try:
        res = requests.get(f"{BASE_URL}/api/cases")
        if res.status_code == 200:
            print("✅ GET /api/cases successful")
        else:
            print(f"❌ GET /api/cases failed: {res.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Note: We need actual files to test uploads properly.
    # This script mainly verifies the server is up and reachable.
    print("API is reachable.")

if __name__ == "__main__":
    test_api()
