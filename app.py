# In your terminal, first run:
# pip install xai-sdk

import os
import json
from typing import List, Dict

from xai_sdk import Client
from xai_sdk.chat import user, system

# Initialize the client
client = Client(api_key=os.getenv("XAI_API_KEY"))


def generate_reply_suggestions(conversation_history: List[Dict[str, str]]) -> List[str]:
    """
    Generate three reply suggestions based on conversation history.
    
    Args:
        conversation_history: List of messages with 'sender' and 'message' keys
    
    Returns:
        List of three suggested replies
    """
    # Create a new chat instance
    chat = client.chat.create(model="grok-4")
    
    # System prompt that instructs Grok to generate structured output
    system_prompt = """You are a helpful AI assistant that analyzes message conversations and suggests appropriate replies.

When given a conversation history, you will provide exactly 3 reply suggestions that:
1. Are contextually appropriate and natural
2. Vary in tone and approach (e.g., friendly, professional, casual)
3. Are concise but complete responses
4. Take into account the entire conversation context

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{
    "suggestions": [
        {
            "reply": "Your first suggested reply here",
            "tone": "friendly/professional/casual/etc"
        },
        {
            "reply": "Your second suggested reply here",
            "tone": "friendly/professional/casual/etc"
        },
        {
            "reply": "Your third suggested reply here",
            "tone": "friendly/professional/casual/etc"
        }
    ]
}

Do not include any other text or explanation, just the JSON object."""
    
    chat.append(system(system_prompt))
    
    # Format the conversation history for the AI
    conversation_text = "Here is the conversation history:\n\n"
    for msg in conversation_history:
        conversation_text += f"{msg['sender']}: {msg['message']}\n"
    
    conversation_text += "\nPlease provide 3 reply suggestions for the last message."
    
    chat.append(user(conversation_text))
    
    # Get the response
    response = chat.sample()
    
    try:
        # Parse the JSON response
        result = json.loads(response.content)
        suggestions = [item['reply'] for item in result['suggestions']]
        return suggestions
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response.content}")
        # Fallback suggestions if parsing fails
        return [
            "I need to think about this and get back to you.",
            "Thanks for reaching out. Let me consider this.",
            "That's interesting. Could you tell me more?"
        ]


def demo_conversation():
    """Run a demo of the reply suggestion feature."""
    
    print("=== Message Reply Suggestion Demo ===\n")
    
    # Example conversation 1: Professional context
    print("Example 1: Professional Email Context")
    print("-" * 40)
    
    conversation1 = [
        {"sender": "Manager", "message": "Hi, I wanted to check in on the project status. How are things progressing?"},
        {"sender": "You", "message": "Hi! The project is on track. We've completed the initial design phase."},
        {"sender": "Manager", "message": "Great to hear! When do you expect to have the prototype ready for review?"}
    ]
    
    print("Conversation:")
    for msg in conversation1:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating reply suggestions...")
    suggestions1 = generate_reply_suggestions(conversation1)
    
    print("\nSuggested replies:")
    for i, suggestion in enumerate(suggestions1, 1):
        print(f"{i}. {suggestion}")
    
    print("\n" + "="*50 + "\n")
    
    # Example conversation 2: Casual friend context
    print("Example 2: Casual Friend Conversation")
    print("-" * 40)
    
    conversation2 = [
        {"sender": "Friend", "message": "Hey! Haven't heard from you in a while. How have you been?"},
        {"sender": "You", "message": "I've been super busy with work, but doing well! How about you?"},
        {"sender": "Friend", "message": "Same here! We should catch up sometime. Are you free this weekend?"}
    ]
    
    print("Conversation:")
    for msg in conversation2:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating reply suggestions...")
    suggestions2 = generate_reply_suggestions(conversation2)
    
    print("\nSuggested replies:")
    for i, suggestion in enumerate(suggestions2, 1):
        print(f"{i}. {suggestion}")
    
    print("\n" + "="*50 + "\n")
    
    # Example conversation 3: Customer service context
    print("Example 3: Customer Service Context")
    print("-" * 40)
    
    conversation3 = [
        {"sender": "Customer", "message": "I ordered a product last week but haven't received it yet."},
        {"sender": "You", "message": "I apologize for the delay. Let me look into your order right away."},
        {"sender": "Customer", "message": "My order number is #12345. When can I expect to receive it?"}
    ]
    
    print("Conversation:")
    for msg in conversation3:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating reply suggestions...")
    suggestions3 = generate_reply_suggestions(conversation3)
    
    print("\nSuggested replies:")
    for i, suggestion in enumerate(suggestions3, 1):
        print(f"{i}. {suggestion}")


def interactive_mode():
    """Run an interactive mode where users can input their own conversations."""
    
    print("\n=== Interactive Mode ===")
    print("Enter a conversation (type 'done' when finished)")
    print("Format: sender: message")
    print("Example: Friend: Hey, want to grab lunch?")
    print("-" * 40)
    
    conversation = []
    
    while True:
        line = input("> ")
        if line.lower() == 'done':
            break
        
        try:
            sender, message = line.split(":", 1)
            conversation.append({
                "sender": sender.strip(),
                "message": message.strip()
            })
        except ValueError:
            print("Invalid format. Please use 'sender: message'")
    
    if conversation:
        print("\nGenerating reply suggestions...")
        suggestions = generate_reply_suggestions(conversation)
        
        print("\nSuggested replies:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("No conversation entered.")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("XAI_API_KEY"):
        print("Error: Please set your XAI_API_KEY environment variable")
        print("You can do this by running: export XAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Run the demo
    try:
        demo_conversation()
        
        # Ask if user wants to try interactive mode
        print("\nWould you like to try interactive mode? (yes/no)")
        if input("> ").lower() in ['yes', 'y']:
            interactive_mode()
            
    except Exception as e:
        print(f"An error occurred: {e}")