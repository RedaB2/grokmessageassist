# Web interface for Message Reply Suggestions
# pip install flask xai-sdk

import os
import json
from flask import Flask, render_template, request, jsonify
from typing import List, Dict

from xai_sdk import Client
from xai_sdk.chat import user, system

app = Flask(__name__)

# Initialize the XAI client
client = Client(api_key=os.getenv("XAI_API_KEY"))


def generate_reply_suggestions(conversation_history: List[Dict[str, str]], user_name: str = "You") -> Dict:
    """Generate reply suggestions and return them with metadata."""
    try:
        # Validate that the last message is NOT from the user
        if not conversation_history:
            return {
                "success": False,
                "error": "No conversation history provided"
            }
        
        last_message = conversation_history[-1]
        if last_message['sender'] == user_name:
            return {
                "success": False,
                "error": "Cannot generate suggestions when you sent the last message. Wait for a reply first."
            }
        
        # Create a new chat instance
        chat = client.chat.create(model="grok-4")
        
        # System prompt for structured output
        system_prompt = f"""You are a helpful AI assistant that analyzes message conversations and suggests appropriate replies.

IMPORTANT CONTEXT:
- The person needing reply suggestions is "{user_name}"
- You should generate replies that "{user_name}" would send to respond to the last message
- The last message in the conversation is from someone else TO "{user_name}"
- Generate replies from the perspective of "{user_name}"

When given a conversation history, provide exactly 3 reply suggestions with analysis.

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{{
    "suggestions": [
        {{
            "reply": "Your first suggested reply here",
            "tone": "friendly/professional/casual/empathetic/assertive",
            "confidence": 0.85,
            "explanation": "Brief explanation of why this is appropriate"
        }},
        {{
            "reply": "Your second suggested reply here",
            "tone": "friendly/professional/casual/empathetic/assertive",
            "confidence": 0.75,
            "explanation": "Brief explanation of why this is appropriate"
        }},
        {{
            "reply": "Your third suggested reply here",
            "tone": "friendly/professional/casual/empathetic/assertive",
            "confidence": 0.70,
            "explanation": "Brief explanation of why this is appropriate"
        }}
    ],
    "conversation_analysis": {{
        "summary": "Brief summary of the conversation context",
        "last_message_intent": "The intent of the last message sent TO {user_name}",
        "suggested_action": "What action {user_name}'s reply should take"
    }}
}}

Guidelines:
1. Each suggestion should be a reply that {user_name} would send
2. Consider that {user_name} is responding to {last_message['sender']}'s message
3. Each suggestion should have a different tone and approach
4. Confidence scores should be between 0 and 1
5. Provide clear, actionable suggestions
6. Consider the full conversation context"""
        
        chat.append(system(system_prompt))
        
        # Format the conversation history
        conversation_text = "Here is the conversation history:\n\n"
        for msg in conversation_history:
            conversation_text += f"{msg['sender']}: {msg['message']}\n"
        
        conversation_text += f"\nPlease provide 3 reply suggestions for {user_name} to respond to {last_message['sender']}'s last message."
        
        chat.append(user(conversation_text))
        
        # Get the response
        response = chat.sample()
        
        # Parse the JSON response
        result = json.loads(response.content)
        return {
            "success": True,
            "data": result
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response.content}")
        return {
            "success": False,
            "error": "Failed to parse AI response",
            "data": {
                "suggestions": [
                    {
                        "reply": "I'll look into this and get back to you.",
                        "tone": "professional",
                        "confidence": 0.7,
                        "explanation": "Safe default response"
                    }
                ],
                "conversation_analysis": {
                    "summary": "Error analyzing conversation",
                    "last_message_intent": "Unknown",
                    "suggested_action": "Acknowledge and investigate"
                }
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.route('/')
def index():
    """Render the main page for the user."""
    return render_template('index.html', is_friend_view=False)


@app.route('/friend')
def friend_view():
    """Render the friend's view for demo purposes."""
    return render_template('index.html', is_friend_view=True)


@app.route('/suggest', methods=['POST'])
def suggest():
    """API endpoint to get reply suggestions."""
    try:
        data = request.json
        conversation_history = data.get('conversation', [])
        user_name = data.get('user_name', 'You')
        
        if not conversation_history:
            return jsonify({
                "success": False,
                "error": "No conversation provided"
            })
        
        # Generate suggestions
        result = generate_reply_suggestions(conversation_history, user_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/examples')
def examples():
    """Return example conversations."""
    examples = [
        {
            "title": "Job Interview Follow-up",
            "conversation": [
                {"sender": "Recruiter", "message": "Thank you for coming in today for the interview!"},
                {"sender": "You", "message": "Thank you for having me! I really enjoyed learning about the role."},
                {"sender": "Recruiter", "message": "We were impressed with your background. We'll be making decisions by Friday. Do you have any questions?"}
            ]
        },
        {
            "title": "Customer Support",
            "conversation": [
                {"sender": "Customer", "message": "I ordered a product last week but haven't received it yet."},
                {"sender": "You", "message": "I apologize for the delay. Let me look into your order right away."},
                {"sender": "Customer", "message": "My order number is #12345. When can I expect to receive it?"}
            ]
        },
        {
            "title": "Team Collaboration",
            "conversation": [
                {"sender": "Colleague", "message": "Hey, are you available to review the presentation?"},
                {"sender": "You", "message": "Sure! I can take a look at it."},
                {"sender": "Colleague", "message": "Great! I've shared it with you. Could you provide feedback by tomorrow?"}
            ]
        }
    ]
    return jsonify(examples)


# Shared conversation storage for demo purposes
shared_conversation = []

@app.route('/sync_conversation', methods=['GET', 'POST'])
def sync_conversation():
    """Sync conversation between user and friend views."""
    global shared_conversation
    
    if request.method == 'POST':
        data = request.json
        shared_conversation = data.get('conversation', [])
        return jsonify({"success": True, "conversation": shared_conversation})
    else:
        return jsonify({"conversation": shared_conversation})


if __name__ == '__main__':
    # Check if API key is set
    if not os.getenv("XAI_API_KEY"):
        print("Error: Please set your XAI_API_KEY environment variable")
        print("You can do this by running: export XAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Message Reply Suggestion Web App...")
    print("\n🚀 Access the app at:")
    print("   User view: http://localhost:5000")
    print("   Friend view: http://localhost:5000/friend")
    print("\nOpen both URLs in different browser windows to simulate a conversation!")
    
    app.run(debug=True, port=5000) 