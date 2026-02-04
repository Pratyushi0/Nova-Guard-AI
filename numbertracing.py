import asyncio
from truecallerpy import login
import sys

async def start_login():
    try:
        phone = "+919696699295" 
        print(f"Requesting OTP for {phone}...")
        
        response = await login(phone)
        
        print("--- Server Response ---")
        print(response) 
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(start_login())
    except KeyboardInterrupt:
        pass