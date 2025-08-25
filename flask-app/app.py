from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import dotenv
dotenv.load_dotenv()
from typing import List, Dict, Tuple
import os
from modellingcongress.inference import predict_action_from_seq
from urllib.parse import quote, unquote


app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production!
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
class CongressionalActionPredictor:
    """
    Wrapper class for your PyTorch model that predicts congressional actions.
    Replace this with your actual model implementation.
    """
    def __init__(self, model=None, inference_dir=None):
        # Load your model here
        # self.model = torch.load(model_path) if model_path else None
        self.model=model
        self.inference_dir=inference_dir
        # Load action vocabulary - mapping from descriptions to any additional metadata
        if inference_dir and os.path.exists(inference_dir):
            with open(os.path.join(inference_dir,"generics.json"), 'r') as f:
                self.action_descriptions = json.load(f)
        else:
            # Placeholder vocabulary - list of unique action descriptions
            self.action_descriptions = [
                "Bill Introduction - Healthcare Reform Act",
                "Committee Hearing - Ways and Means Committee",
                "Floor Vote - Infrastructure Investment Bill",
                "Amendment Proposal - Tax Code Section 501",
                "Subcommittee Review - Environmental Policy",
                "Conference Committee - Defense Authorization",
                "Final Passage - Education Funding Bill",
                "Presidential Veto - Immigration Reform Act",
                "Override Vote - Climate Change Legislation",
                "Markup Session - Social Security Reform",
                "Bill Introduction - Veterans Affairs Reform",
                "Committee Hearing - Judiciary Committee",
                "Floor Vote - Trade Agreement Ratification",
                "Amendment Proposal - Healthcare Expansion",
                "Subcommittee Review - Technology Policy",
                "Conference Committee - Budget Reconciliation",
                "Final Passage - Criminal Justice Reform",
                "Presidential Signature - Infrastructure Bill",
                "Override Vote - Tax Reform Legislation",
                "Markup Session - Agricultural Policy",
                "Bill Introduction - Clean Energy Investment",
                "Committee Hearing - Foreign Relations Committee",
                "Floor Vote - Minimum Wage Increase",
                "Amendment Proposal - Privacy Protection Act",
                "Subcommittee Review - Financial Services",
                "Conference Committee - Transportation Authorization",
                "Final Passage - Student Loan Reform",
                "Presidential Veto - Corporate Tax Changes",
                "Override Vote - Voting Rights Protection",
                "Markup Session - Housing Policy Reform"
            ]
    
    def predict_next_actions(self, action_sequence: List[str]) -> List[Tuple[str, float]]:
        """
        Given a sequence of congressional actions, predict probabilities for next actions.
        
        Args:
            action_sequence: List of action descriptions representing the sequence so far
            
        Returns:
            List of tuples: (action_description, probability)
        """
        if action_sequence and action_sequence[-1]=="No further actions.":
            return []
        return list(predict_action_from_seq(self.model,action_sequence,self.inference_dir)[0].items())
        # Actual model inference code would go here:
        # You'll need to implement a mapping between action descriptions and model indices
        # For example:
        # 
        # # Convert descriptions to indices for model input
        # action_indices = [self.description_to_index[desc] for desc in action_sequence]
        # 
        # with torch.no_grad():
        #     input_tensor = torch.tensor(action_indices).unsqueeze(0).to(self.device)
        #     logits = self.model(input_tensor)
        #     probabilities = F.softmax(logits[:, -1, :], dim=-1)
        #     
        #     predictions = []
        #     for idx, prob in enumerate(probabilities[0]):
        #         if idx in self.index_to_description:
        #             description = self.index_to_description[idx]
        #             predictions.append((description, prob.item()))
        #     
        #     predictions.sort(key=lambda x: x[1], reverse=True)
        #     return predictions
        
        return predictions
    
    def is_valid_action(self, action_description: str) -> bool:
        """Check if an action description is valid"""
        return action_description in self.action_descriptions

# Initialize the predictor
predictor = CongressionalActionPredictor("lr3e-04_lassoweight1e-07_batch256",inference_dir="inference") 


def get_current_sequence():
    """Get the current action sequence from session"""
    sequence = session.get('action_sequence', [])
    # Handle legacy data or mixed types by converting everything to strings
    clean_sequence = []
    for item in sequence:
        if isinstance(item, str):
            clean_sequence.append(item)
        else:
            # Handle any other unexpected types
            clean_sequence.append(str(item))
    return clean_sequence

def set_current_sequence(sequence):
    """Set the current action sequence in session"""
    # Ensure all items in sequence are strings
    clean_sequence = [str(item) for item in sequence]
    session['action_sequence'] = clean_sequence

def filter_predictions(predictions, search_query):
    """Filter predictions based on search query"""
    if not search_query:
        return predictions
    
    search_lower = search_query.lower()
    return [
        (description, prob) for description, prob in predictions
        if search_lower in description.lower()
    ]

def encode_action_description(description):
    """URL-encode action description for safe use in URLs"""
    return quote(description, safe='')

def decode_action_description(encoded_description):
    """URL-decode action description from URL"""
    return unquote(encoded_description)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with the interactive tool"""
    search_query = ''
    error_message = None
    
    if request.method == 'POST':
        search_query = request.form.get('search', '').strip()
    
    try:
        # Get current sequence
        action_sequence = get_current_sequence()
        
        # Get predictions from the model
        all_predictions = predictor.predict_next_actions(action_sequence)
        
        # Filter predictions based on search query
        filtered_predictions = sorted(filter_predictions(all_predictions, search_query), key=lambda x:x[1],reverse=True)
        # Format predictions for display
        predictions_display = [
            {
                'description': description,
                'encoded_description': encode_action_description(description),
                'probability': round(prob * 100, 2)
            }
            for description, prob in filtered_predictions
        ]
        
        # Get sequence with step numbers
        sequence_display = [
            {
                'step': i + 1,
                'description': description
            }
            for i, description in enumerate(action_sequence)
        ]
        # Calculate statistics
        stats = {
            'sequence_length': len(action_sequence),
            'total_actions': len(all_predictions),
            'filtered_predictions': len(filtered_predictions),
            'top_probability': round(filtered_predictions[0][1] * 100, 2) if filtered_predictions else 0
        }
        
        return render_template('index.html',
                             predictions=predictions_display,
                             sequence=sequence_display,
                             stats=stats,
                             search_query=search_query,
                             error_message=error_message)
    
    except Exception as e:
        error_message = f"Error loading predictions: {str(e)}"
        return render_template('index.html',
                             predictions=[],
                             sequence=[],
                             stats={'sequence_length': 0, 'total_actions': 0, 'filtered_predictions': 0, 'top_probability': 0},
                             search_query=search_query,
                             error_message=error_message)

@app.route('/select_action/<path:encoded_description>')
def select_action(encoded_description):
    """Add an action to the sequence"""
    try:
        # Decode the action description
        action_description = decode_action_description(encoded_description)
        
        # Validate the action
        if not predictor.is_valid_action(action_description):
            flash(f"Invalid action: {action_description}", 'error')
            return redirect(url_for('index'))
        
        # Get current sequence and add new action
        action_sequence = get_current_sequence()
        action_sequence.append(action_description)
        set_current_sequence(action_sequence)
        
        flash(f"Added: {action_description}", 'success')
        
    except Exception as e:
        flash(f"Error adding action: {str(e)}", 'error')
    
    return redirect(url_for('index'))

@app.route('/undo_action')
def undo_action():
    """Remove the last action from the sequence"""
    try:
        action_sequence = get_current_sequence()
        if action_sequence:
            removed_action = action_sequence.pop()
            set_current_sequence(action_sequence)
            flash(f"Removed: {removed_action}", 'info')
        else:
            flash("No actions to undo", 'warning')
            
    except Exception as e:
        flash(f"Error undoing action: {str(e)}", 'error')
    
    return redirect(url_for('index'))

@app.route('/reset_sequence')
def reset_sequence():
    """Clear the entire action sequence"""
    try:
        # Clear the session completely to avoid any legacy data issues
        session.pop('action_sequence', None)
        set_current_sequence([])
        flash("Sequence reset", 'info')
        
    except Exception as e:
        flash(f"Error resetting sequence: {str(e)}", 'error')
    
    return redirect(url_for('index'))

@app.route('/action_info/<path:encoded_description>')
def action_info(encoded_description):
    """Get detailed information about a specific action"""
    try:
        action_description = decode_action_description(encoded_description)
        
        if predictor.is_valid_action(action_description):
            flash(f"Action: {action_description}", 'info')
        else:
            flash(f"Action not found: {action_description}", 'error')
            
    except Exception as e:
        flash(f"Error retrieving action info: {str(e)}", 'error')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5001)