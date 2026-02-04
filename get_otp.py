import asyncio
from truecallerpy import login

async def start_login():
    # Replace with your actual phone number
    phone = "+919696699295"  
    print(f"Requesting OTP for {phone}...")
    
    try:
        response = await login(phone)
        print("\n--- Server Response ---")
        print(response) 
        
        if response.get("status") == 1 or response.get("message") == "Sent":
            print("\n✅ Success! Check your SMS or Truecaller App for the OTP.")
        elif response.get("status") == 6:
            print("\n❌ Error: Too many attempts. Wait 24 hours.")
        else:
            print("\n❓ Something went wrong. Check the response above.")
            
    except Exception as e:
        print(f"\n⚠️ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(start_login())
