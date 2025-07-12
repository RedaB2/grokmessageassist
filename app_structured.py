# Enhanced version with Pydantic models for structured outputs
# pip install xai-sdk pydantic

import os
import json
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

from xai_sdk import Client
from xai_sdk.chat import user, system

# Initialize the client
client = Client(api_key=os.getenv("XAI_API_KEY"))


# Define Pydantic models for structured outputs
class ReplySuggestion(BaseModel):
    """Model for a single reply suggestion"""
    reply: str = Field(description="The suggested reply text")
    tone: Literal["friendly", "professional", "casual", "empathetic", "assertive"] = Field(
        description="The tone of the reply"
    )
    confidence: float = Field(
        description="Confidence score for this suggestion (0-1)",
        ge=0.0, 
        le=1.0
    )
    context_notes: Optional[str] = Field(
        description="Brief notes about why this reply is appropriate",
        default=None
    )


class ReplyResponses(BaseModel):
    """Model for the complete response containing multiple suggestions"""
    suggestions: List[ReplySuggestion] = Field(
        description="List of reply suggestions",
        min_items=3,
        max_items=3
    )
    conversation_summary: str = Field(
        description="Brief summary of the conversation context"
    )
    primary_intent: str = Field(
        description="The primary intent detected in the last message"
    )


def generate_structured_reply_suggestions(
    conversation_history: List[Dict[str, str]], 
    context: Optional[str] = None
) -> ReplyResponses:
    """
    Generate structured reply suggestions with metadata.
    
    Args:
        conversation_history: List of messages with 'sender' and 'message' keys
        context: Optional additional context about the conversation
    
    Returns:
        ReplyResponses object with suggestions and metadata
    """
    # Create a new chat instance
    chat = client.chat.create(model="grok-4")
    
    # Create the schema from our Pydantic model
    schema = ReplyResponses.model_json_schema()
    
    # System prompt with schema
    system_prompt = f"""You are a helpful AI assistant that analyzes message conversations and suggests appropriate replies.

You must analyze the conversation and provide exactly 3 reply suggestions with metadata.

Your response must be a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Guidelines for suggestions:
1. Each suggestion should have a different tone and approach
2. Consider the full conversation context
3. Provide confidence scores based on how well the suggestion fits
4. Include brief context notes explaining why each suggestion is appropriate
5. Summarize the conversation and identify the primary intent

IMPORTANT: Respond ONLY with a valid JSON object. No additional text."""
    
    chat.append(system(system_prompt))
    
    # Format the conversation history
    conversation_text = "Conversation history:\n\n"
    for msg in conversation_history:
        conversation_text += f"{msg['sender']}: {msg['message']}\n"
    
    if context:
        conversation_text += f"\nAdditional context: {context}\n"
    
    conversation_text += "\nGenerate 3 reply suggestions with metadata."
    
    chat.append(user(conversation_text))
    
    # Get the response
    response = chat.sample()
    
    try:
        # Parse and validate with Pydantic
        result_data = json.loads(response.content)
        return ReplyResponses(**result_data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response.content}")
        
        # Return default structured response
        return ReplyResponses(
            suggestions=[
                ReplySuggestion(
                    reply="I'll need to think about this and get back to you.",
                    tone="professional",
                    confidence=0.7,
                    context_notes="Safe default response"
                ),
                ReplySuggestion(
                    reply="Thanks for reaching out. Let me look into this.",
                    tone="friendly",
                    confidence=0.7,
                    context_notes="Polite acknowledgment"
                ),
                ReplySuggestion(
                    reply="Could you provide more details about this?",
                    tone="casual",
                    confidence=0.6,
                    context_notes="Information gathering"
                )
            ],
            conversation_summary="Unable to analyze conversation",
            primary_intent="Unknown"
        )


def display_structured_suggestions(response: ReplyResponses):
    """Display the structured suggestions in a formatted way."""
    print(f"\n📊 Conversation Analysis")
    print(f"Summary: {response.conversation_summary}")
    print(f"Primary Intent: {response.primary_intent}")
    print("\n💬 Reply Suggestions:")
    print("-" * 60)
    
    for i, suggestion in enumerate(response.suggestions, 1):
        print(f"\nOption {i}:")
        print(f"Reply: \"{suggestion.reply}\"")
        print(f"Tone: {suggestion.tone}")
        print(f"Confidence: {suggestion.confidence:.0%}")
        if suggestion.context_notes:
            print(f"Notes: {suggestion.context_notes}")


def demo_structured_conversations():
    """Run demos with structured outputs."""
    
    print("=== Enhanced Message Reply Suggestion Demo ===")
    print("Using structured outputs with metadata\n")
    
    # Example 1: Job interview follow-up
    print("Example 1: Job Interview Follow-up")
    print("-" * 60)
    
    conversation1 = [
        {"sender": "Recruiter", "message": "Thank you for coming in today for the interview!"},
        {"sender": "You", "message": "Thank you for having me! I really enjoyed learning about the role."},
        {"sender": "Recruiter", "message": "We were impressed with your background. We'll be making decisions by Friday. Do you have any questions in the meantime?"}
    ]
    
    print("Conversation:")
    for msg in conversation1:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating structured suggestions...")
    response1 = generate_structured_reply_suggestions(
        conversation1,
        context="Job interview for software engineer position"
    )
    display_structured_suggestions(response1)
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Difficult customer complaint
    print("Example 2: Customer Complaint Resolution")
    print("-" * 60)
    
    conversation2 = [
        {"sender": "Customer", "message": "This is the third time I've had to contact you about this issue!"},
        {"sender": "You", "message": "I sincerely apologize for the frustration this has caused you."},
        {"sender": "Customer", "message": "I want a full refund and compensation for my time. This is unacceptable!"}
    ]
    
    print("Conversation:")
    for msg in conversation2:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating structured suggestions...")
    response2 = generate_structured_reply_suggestions(
        conversation2,
        context="Customer has had repeated technical issues with premium subscription"
    )
    display_structured_suggestions(response2)
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Team collaboration
    print("Example 3: Team Project Coordination")
    print("-" * 60)
    
    conversation3 = [
        {"sender": "Teammate", "message": "Hey, I noticed the deadline got moved up to next Tuesday"},
        {"sender": "You", "message": "Oh wow, that's much sooner than expected!"},
        {"sender": "Teammate", "message": "Yeah, the client wants to see a demo. Should we meet to replan our approach?"}
    ]
    
    print("Conversation:")
    for msg in conversation3:
        print(f"{msg['sender']}: {msg['message']}")
    
    print("\nGenerating structured suggestions...")
    response3 = generate_structured_reply_suggestions(conversation3)
    display_structured_suggestions(response3)


def save_suggestions_to_file(response: ReplyResponses, filename: str = "suggestions.json"):
    """Save the structured suggestions to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(response.model_dump(), f, indent=2)
    print(f"\n💾 Suggestions saved to {filename}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("XAI_API_KEY"):
        print("Error: Please set your XAI_API_KEY environment variable")
        print("You can do this by running: export XAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Run the structured demo
    try:
        demo_structured_conversations()
        
        # Example of saving suggestions
        print("\n" + "="*80)
        print("Generating suggestions to save...")
        
        example_conversation = [
            {"sender": "Client", "message": "Can we schedule a meeting to discuss the project updates?"},
            {"sender": "You", "message": "Of course! I'd be happy to discuss the progress."},
            {"sender": "Client", "message": "Great! Are you available tomorrow afternoon or Thursday morning?"}
        ]
        
        response = generate_structured_reply_suggestions(example_conversation)
        save_suggestions_to_file(response)
        
    except Exception as e:
        print(f"An error occurred: {e}") 