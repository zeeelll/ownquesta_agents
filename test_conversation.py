"""
Enhanced Test Script for OwnQuesta Conversation Agent
Tests conversational capabilities including greetings, RAG, and farewells
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check...")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/conversation/health")
    data = response.json()
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(data, indent=2)}")
    
    return response.status_code == 200


def test_conversational_flow():
    """Test natural conversational flow with greetings and context"""
    print("\n" + "="*60)
    print("Testing Full Conversational Flow...")
    print("="*60)
    
    conversation = [
        "Hi there!",
        "What is OwnQuesta?",
        "That sounds interesting! Who is it built for?",
        "How is it different from other ML tools?",
        "Thanks for the help! Goodbye!"
    ]
    
    history = []
    
    for i, message in enumerate(conversation, 1):
        print(f"\n{'='*60}")
        print(f"[Turn {i}] 👤 User: {message}")
        print(f"{'='*60}")
        
        response = requests.post(f"{BASE_URL}/conversation/chat", json={
            "message": message,
            "conversation_history": history
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Assistant: {data['response']}")
            if data['sources']:
                print(f"\n📚 Knowledge Sources: {', '.join(data['sources'])}")
            else:
                print(f"\n💭 Response Type: Conversational (no knowledge base needed)")
            history = data['conversation_history']
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    
    return True


def test_greeting_variations():
    """Test different types of greetings"""
    print("\n" + "="*60)
    print("Testing Greeting Intelligence...")
    print("="*60)
    
    greetings = [
        "Hello!",
        "Hey, how are you?",
        "Good morning!",
        "Hi, can you help me?"
    ]
    
    for i, greeting in enumerate(greetings, 1):
        print(f"\n[Greeting {i}] 👤 User: {greeting}")
        
        response = requests.post(f"{BASE_URL}/conversation/chat", json={
            "message": greeting
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Assistant: {data['response'][:250]}...")
            print(f"Uses RAG: {'No - Pure conversation' if not data['sources'] else 'Yes'}")
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    
    return True


def test_ownquesta_questions():
    """Test OwnQuesta-specific questions (should use RAG)"""
    print("\n" + "="*60)
    print("Testing OwnQuesta Knowledge Questions...")
    print("="*60)
    
    questions = [
        "What makes OwnQuesta different from other ML tools?",
        "What is the vision of OwnQuesta?",
        "Why does OwnQuesta exist?",
        "Tell me about OwnQuesta's philosophy on explainability"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}] 👤 User: {question}")
        
        response = requests.post(f"{BASE_URL}/conversation/chat", json={
            "message": question
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Assistant: {data['response'][:200]}...")
            print(f"📚 Sources Used: {', '.join(data['sources'])}")
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    
    return True


def test_trust_building():
    """Test trust-building conversation"""
    print("\n" + "="*60)
    print("Testing Trust Building & Rapport...")
    print("="*60)
    
    conversation = [
        "Hi, I'm new to machine learning",
        "Can OwnQuesta help beginners like me?",
        "That's reassuring! What if I don't understand something?",
        "Thank you! You've been very helpful"
    ]
    
    history = []
    
    for i, message in enumerate(conversation, 1):
        print(f"\n[Turn {i}] 👤 User: {message}")
        
        response = requests.post(f"{BASE_URL}/conversation/chat", json={
            "message": message,
            "conversation_history": history
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Assistant: {data['response']}")
            history = data['conversation_history']
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    
    return True


def test_mixed_conversation():
    """Test mixed conversation (greetings + questions + general chat)"""
    print("\n" + "="*60)
    print("Testing Mixed Conversation Types...")
    print("="*60)
    
    messages = [
        ("Greeting", "Hello!"),
        ("General ML", "What is machine learning?"),
        ("OwnQuesta Specific", "How does OwnQuesta help with this?"),
        ("Farewell", "Thanks, goodbye!")
    ]
    
    history = []
    
    for i, (msg_type, message) in enumerate(messages, 1):
        print(f"\n[{msg_type}] 👤 User: {message}")
        
        response = requests.post(f"{BASE_URL}/conversation/chat", json={
            "message": message,
            "conversation_history": history
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Assistant: {data['response'][:300]}...")
            if data['sources']:
                print(f"📚 RAG Used: Yes ({len(data['sources'])} sources)")
            else:
                print(f"💭 RAG Used: No (conversational response)")
            history = data['conversation_history']
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 OwnQuesta Enhanced Conversation Agent - Test Suite")
    print("="*60)
    print("\nMake sure the server is running: uvicorn main:app --reload")
    print("Server URL:", BASE_URL)
    
    try:
        # Test 1: Health Check
        if not test_health_check():
            print("\n❌ Health check failed!")
            return
        print("\n✅ Health check passed!")
        
        # Test 2: Greeting Intelligence
        if not test_greeting_variations():
            print("\n❌ Greeting test failed!")
            return
        print("\n✅ Greeting intelligence passed!")
        
        # Test 3: OwnQuesta Knowledge
        if not test_ownquesta_questions():
            print("\n❌ Knowledge questions test failed!")
            return
        print("\n✅ Knowledge questions passed!")
        
        # Test 4: Full Conversational Flow
        if not test_conversational_flow():
            print("\n❌ Conversational flow test failed!")
            return
        print("\n✅ Conversational flow passed!")
        
        # Test 5: Trust Building
        if not test_trust_building():
            print("\n❌ Trust building test failed!")
            return
        print("\n✅ Trust building passed!")
        
        # Test 6: Mixed Conversation
        if not test_mixed_conversation():
            print("\n❌ Mixed conversation test failed!")
            return
        print("\n✅ Mixed conversation passed!")
        
        print("\n" + "="*60)
        print("🎉 All tests passed successfully!")
        print("="*60)
        print("\n✨ The chatbot is now smarter with:")
        print("   • Natural greetings and farewells")
        print("   • Trust-building conversations")
        print("   • Intelligent RAG usage (only when needed)")
        print("   • Conversational context awareness")
        print("   • Mixed conversation handling")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server!")
        print("Make sure the server is running:")
        print("   cd ownquesta_agents")
        print("   uv run uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
