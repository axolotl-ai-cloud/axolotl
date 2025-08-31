#!/usr/bin/env python3
"""
Interactive Golf Strategy Chat
Chat with your trained model to get golf strategy recommendations.

Usage:
    python golf_strategy_chat.py

Example conversation:
    You: What are the strategies for hole 12?
    Bot: [Lists available strategies for hole 12]
    You: I can drive it 250 yards, which strategy should I use?
    Bot: Strategy: 155 yards
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

class GolfStrategyChat:
    def __init__(self, adapter_path="outputs/bethpage-lora/checkpoint-8_hour_enhanced"):
        self.adapter_path = adapter_path
        self.hole_data = {}
        self.current_hole = None
        self.load_model()
        self.load_hole_data()
        
    def load_model(self):
        """Load the trained LoRA model"""
        print("Loading model and tokenizer...")
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
    def load_hole_data(self):
        """Load hole information from training data"""
        data_file = "data/bethpage_black/train_strategy_logic.jsonl"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                for line in f:
                    example = json.loads(line.strip())
                    hole = example.get('hole')
                    strategies = example.get('tee_shot_strategies', [])
                    description = example.get('description', '').strip()
                    
                    if hole and strategies:  # Only process if hole has strategies
                        if hole not in self.hole_data:
                            self.hole_data[hole] = {
                                'par': example.get('par'),
                                'yardage': example.get('yardage'),
                                'descriptions': set(),  # Use set to avoid duplicates
                                'strategies': []
                            }
                        
                        # Collect unique descriptions
                        if description and len(description) > 10:  # Filter out very short descriptions
                            self.hole_data[hole]['descriptions'].add(description)
                        
                        # Merge strategies, avoiding duplicates
                        existing_cutoffs = {s.get('cutoff_distance') for s in self.hole_data[hole]['strategies']}
                        for strategy in strategies:
                            cutoff = strategy.get('cutoff_distance')
                            if cutoff and cutoff not in existing_cutoffs:
                                self.hole_data[hole]['strategies'].append(strategy)
                                existing_cutoffs.add(cutoff)
        
        # Convert description sets to lists for easier handling
        for hole_data in self.hole_data.values():
            hole_data['descriptions'] = list(hole_data['descriptions'])
                                
        print(f"✅ Loaded data for {len(self.hole_data)} holes")
    
    def extract_hole_number(self, text):
        """Extract hole number from user input"""
        patterns = [
            r"hole\s*(\d+)",
            r"#(\d+)",
            r"(\d+)(?:st|nd|rd|th)?\s*hole",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                hole_num = int(match.group(1))
                if 1 <= hole_num <= 18:
                    return hole_num
        return None
    
    def extract_drive_distance(self, text):
        """Extract drive distance from user input"""
        patterns = [
            r"drive\s*(?:it\s*)?(\d{2,3})\s*yards?",
            r"hit\s*(?:it\s*)?(\d{2,3})\s*yards?",
            r"(\d{2,3})\s*yard\s*drive",
            r"(\d{2,3})\s*yards?",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                distance = int(match.group(1))
                if 150 <= distance <= 400:  # Reasonable drive distances
                    return distance
        return None
    
    def format_hole_info(self, hole_num):
        """Format hole information for display using model-generated description"""
        if hole_num not in self.hole_data:
            return f"❌ Sorry, I don't have data for hole {hole_num}."
        
        hole = self.hole_data[hole_num]
        info = f"🏌️ **Hole {hole_num} - Bethpage Black**\n"
        info += f"Par {hole['par']}, {hole['yardage']} yards\n"
        
        # Get model-generated description
        description = self.get_hole_description(hole_num)
        if description:
            info += f"📝 {description}\n"
        
        if hole['strategies']:
            info += f"\n⛳ **Available Strategies:**\n"
            for i, strategy in enumerate(hole['strategies'], 1):
                cutoff = strategy.get('cutoff_distance')
                remaining = strategy.get('remaining_distance', 'N/A')
                advantage = strategy.get('advantage', '')
                
                info += f"  {i}. **{cutoff} yards** - Leaves {remaining} yards to pin"
                if advantage:
                    info += f" ({advantage})"
                info += "\n"
        
        return info
    
    def get_hole_description(self, hole_num):
        """Generate hole description by synthesizing multiple source descriptions"""
        if hole_num not in self.hole_data:
            return "❌ Sorry, I don't have data for that hole."
        
        hole = self.hole_data[hole_num]
        descriptions = hole.get('descriptions', [])
        
        if not descriptions:
            return f"A challenging hole at Bethpage Black."
        
        if len(descriptions) == 1:
            # If only one description, return it as-is
            return descriptions[0]
        
        # Multiple descriptions - ask model to synthesize them
        prompt = f"Combine these {len(descriptions)} descriptions of Hole {hole_num} at Bethpage Black (Par {hole['par']}, {hole['yardage']} yards) into one cohesive narrative:\n\n"
        
        for i, desc in enumerate(descriptions, 1):
            prompt += f"{i}. {desc}\n\n"
        
        prompt += "Synthesized description:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # If model fails to generate good description, fall back to first available
        if not response or len(response) < 20 or "Strategy:" in response:
            return descriptions[0]  # Fallback to first description
        
        return response

    def get_strategy_recommendation(self, hole_num, drive_distance):
        """Get strategy recommendation using the trained model"""
        if hole_num not in self.hole_data:
            return "❌ Sorry, I don't have data for that hole."
        
        # Create prompt similar to training format
        hole = self.hole_data[hole_num]
        prompt = f"Hole {hole_num}, Bethpage Black: Par {hole['par']}, {hole['yardage']} yards. Golfer drives {drive_distance} yards. Best strategy?"
        prompt += " Answer: Strategy: <N> yards"
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=60)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Extract strategy number
        strategy_match = re.search(r"Strategy:\s*(\d{2,3})\s*yards?", response, re.IGNORECASE)
        if strategy_match:
            strategy_yards = int(strategy_match.group(1))
            
            # Add explanation of choice
            available_cutoffs = [s['cutoff_distance'] for s in hole['strategies'] if isinstance(s.get('cutoff_distance'), int)]
            eligible = [c for c in available_cutoffs if c <= drive_distance]
            
            explanation = f"\n🎯 **Recommended Strategy: {strategy_yards} yards**\n"
            explanation += f"📊 Available strategies: {sorted(available_cutoffs)} yards\n"
            explanation += f"✅ Eligible for {drive_distance}-yard drive: {sorted(eligible) if eligible else ['None - using shortest option']}\n"
            explanation += f"🧠 Logic: Choose highest available ≤ drive distance"
            
            return explanation
        else:
            return f"🤖 Model response: {response}"
    
    def process_input(self, user_input):
        """Process user input and generate appropriate response"""
        user_input = user_input.strip()
        
        if not user_input:
            return "👋 Ask me about golf strategies! Try: 'What are the strategies for hole 12?'"
        
        # Check for hole information request
        hole_num = self.extract_hole_number(user_input)
        drive_distance = self.extract_drive_distance(user_input)
        
        # Handle different types of queries
        if "strategies" in user_input.lower() or "available" in user_input.lower():
            if hole_num:
                self.current_hole = hole_num
                return self.format_hole_info(hole_num)
            else:
                return "🤔 Which hole are you asking about? Try: 'What are the strategies for hole 5?'"
        
        elif "strategy" in user_input.lower() or "should i use" in user_input.lower() or "recommend" in user_input.lower():
            if hole_num and drive_distance:
                self.current_hole = hole_num
                return self.get_strategy_recommendation(hole_num, drive_distance)
            elif drive_distance and self.current_hole:
                return self.get_strategy_recommendation(self.current_hole, drive_distance)
            elif hole_num and not drive_distance:
                self.current_hole = hole_num
                return f"{self.format_hole_info(hole_num)}\n💭 How far can you drive? Try: 'I can drive 250 yards, which strategy should I use?'"
            else:
                return "🤔 I need to know the hole number and your drive distance. Try: 'Hole 5, I drive 280 yards, what strategy?'"
        
        elif drive_distance and self.current_hole:
            return self.get_strategy_recommendation(self.current_hole, drive_distance)
        
        elif hole_num:
            self.current_hole = hole_num
            return self.format_hole_info(hole_num)
        
        else:
            return """🏌️ **Golf Strategy Chat Help**
            
Try these examples:
• "What are the strategies for hole 12?"
• "I can drive 250 yards, which strategy should I use?"
• "Hole 5, I drive 280 yards, what strategy?"
• "Show me hole 3"
• "280 yards" (if we're already discussing a hole)

Currently discussing: """ + (f"Hole {self.current_hole}" if self.current_hole else "None")

def main():
    print("🏌️ Golf Strategy Chat - Bethpage Black")
    print("=" * 50)
    print("Loading your trained model...")
    
    try:
        chat = GolfStrategyChat()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have a trained model at: outputs/bethpage-lora/checkpoint-1_hour")
        return
    
    print("\n✅ Ready to chat! Type 'quit' to exit.")
    print("💡 Try: 'What are the strategies for hole 12?'")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🏌️ You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for using Golf Strategy Chat!")
                break
            
            response = chat.process_input(user_input)
            print(f"\n🤖 Golf Pro: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Thanks for using Golf Strategy Chat!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
