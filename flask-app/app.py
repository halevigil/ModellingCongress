from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import json
from typing import List, Dict, Tuple
import os

app = Flask(__name__)

class CongressionalActionPredictor:
    """
    Wrapper class for your PyTorch model that predicts congressional actions.
    Replace this with your actual model implementation.
    """
    def __init__(self, model_path=None, action_vocab_path=None):
        # Load your model here
        # self.model = torch.load(model_path) if model_path else None
        self.model = None  # Placeholder - replace with your actual model
        
        # Load action vocabulary - mapping from action IDs to human-readable descriptions
        if action_vocab_path and os.path.exists(action_vocab_path):
            with open(action_vocab_path, 'r') as f:
                self.action_vocab = json.load(f)
        else:
            # Placeholder vocabulary - replace with your actual action categories
            self.action_vocab = {
                0: "Bill Introduction - Healthcare Reform",
                1: "Committee Hearing - Budget Committee",
                2: "Floor Vote - Infrastructure Bill",
                3: "Amendment Proposal - Tax Reform",
                4: "Subcommittee Review - Environmental Policy",
                5: "Conference Committee - Defense Authorization",
                6: "Final Passage - Education Funding",
                7: "Presidential Veto - Immigration Reform",
                8: "Override Vote - Climate Change Bill",
                9: "Markup Session - Social Security Reform",
                # Add more actions as needed
            }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def predict_next_actions(self, action_sequence: List[int]) -> List[Tuple[int, float, str]]:
        """
        Given a sequence of congressional actions, predict probabilities for next actions.
        
        Args:
            action_sequence: List of action IDs representing the sequence so far
            
        Returns:
            List of tuples: (action_id, probability, description)
        """
        if not self.model:
            # Placeholder predictions for demo - replace with actual model inference
            import random
            predictions = []
            for action_id, description in self.action_vocab.items():
                prob = random.random()
                predictions.append((action_id, prob, description))
            
            # Sort by probability (highest first)
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions
        
        # Actual model inference code would go here:
        # with torch.no_grad():
        #     input_tensor = torch.tensor(action_sequence).unsqueeze(0).to(self.device)
        #     logits = self.model(input_tensor)
        #     probabilities = F.softmax(logits[:, -1, :], dim=-1)
        #     
        #     predictions = []
        #     for action_id, prob in enumerate(probabilities[0]):
        #         if action_id in self.action_vocab:
        #             predictions.append((action_id, prob.item(), self.action_vocab[action_id]))
        #     
        #     predictions.sort(key=lambda x: x[1], reverse=True)
        #     return predictions
        
        return predictions

# Initialize the predictor
predictor = CongressionalActionPredictor()

@app.route('/')
def index():
    """Main page with the interactive tool"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to get predictions for next actions"""
    try:
        data = request.json
        action_sequence = data.get('sequence', [])
        search_query = data.get('search', '').lower()
        
        # Get predictions from the model
        predictions = predictor.predict_next_actions(action_sequence)
        
        # Filter predictions based on search query if provided
        if search_query:
            filtered_predictions = [
                (action_id, prob, desc) for action_id, prob, desc in predictions
                if search_query in desc.lower()
            ]
        else:
            filtered_predictions = predictions
        
        # Format response
        response = {
            'predictions': [
                {
                    'action_id': action_id,
                    'probability': round(prob * 100, 2),  # Convert to percentage
                    'description': desc
                }
                for action_id, prob, desc in filtered_predictions
            ],
            'sequence_length': len(action_sequence),
            'total_actions': len(predictions)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/action_info/<int:action_id>')
def action_info(action_id):
    """Get detailed information about a specific action"""
    if action_id in predictor.action_vocab:
        return jsonify({
            'action_id': action_id,
            'description': predictor.action_vocab[action_id]
        })
    else:
        return jsonify({'error': 'Action not found'}), 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)