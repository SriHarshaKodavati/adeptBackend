from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient("mongodb+srv://kodavati009:harsha@cluster1.ak54w.mongodb.net/modules?retryWrites=true&w=majority")
db = client['modules']
questions_collection = db['module_questions']

class AdaptiveELearningModel:
    def __init__(self):
        self.user_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.module_skip_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.train_models()
    
    def train_models(self):
        # Training data for user classification - Each row is [score, time_taken]
        X_users = np.array([
            [100, 10], [95, 12], [90, 14], [85, 16], [80, 18],    # Advanced
            [75, 20], [70, 22], [65, 24], [60, 26], [55, 28],     # Advanced/Intermediate
            [50, 30], [45, 32], [40, 34], [35, 36], [30, 38],     # Intermediate
            [25, 40], [20, 42], [15, 44], [10, 46], [5, 48],      # Basic
            [85, 18], [92, 11], [77, 20], [66, 25], [56, 28],     # Mixed
            [46, 32], [36, 35], [26, 40], [16, 45], [6, 50],      # Basic
            [99, 10], [94, 15], [89, 12], [79, 22], [67, 30],     # Advanced/Intermediate
            [55, 34], [45, 38], [34, 40], [24, 45], [14, 48]      # Basic
        ])
        
        y_users = np.array([
            'advanced', 'advanced', 'advanced', 'advanced', 'advanced',           # 5
            'advanced', 'advanced', 'intermediate', 'intermediate', 'intermediate', # 5
            'intermediate', 'intermediate', 'intermediate', 'basic', 'basic',     # 5
            'basic', 'basic', 'basic', 'basic', 'basic',                         # 5
            'advanced', 'advanced', 'advanced', 'intermediate', 'intermediate',   # 5
            'intermediate', 'basic', 'basic', 'basic', 'basic',                  # 5
            'advanced', 'advanced', 'advanced', 'intermediate', 'intermediate',   # 5
            'basic', 'basic', 'basic', 'basic', 'basic'                         # 5
        ])
        
        # Verification of array lengths
        assert len(X_users) == len(y_users), f"Mismatch in training data lengths: X={len(X_users)}, y={len(y_users)}"
        
        # Training data for module skip classification
        X_modules = np.array([
            [100, 10], [95, 12], [90, 14], [85, 16], [80, 18],
            [75, 20], [70, 22], [65, 24], [60, 26], [55, 28],
            [50, 30], [45, 32], [40, 34], [35, 36], [30, 38],
            [25, 40], [20, 42], [15, 44], [10, 46], [5, 48],
            [10, 10], [20, 10], [30, 15], [40, 20], [50, 25],
            [60, 30], [70, 35], [80, 40], [90, 45], [100, 5]
        ])
        
        y_modules = np.array([
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ])
        
        # Verify module data consistency
        assert len(X_modules) == len(y_modules), f"Mismatch in module data lengths: X={len(X_modules)}, y={len(y_modules)}"
        
        # Split and train user classification model
        X_users_train, X_users_test, y_users_train, y_users_test = train_test_split(
            X_users, y_users, test_size=0.2, random_state=42
        )
        
        self.user_classifier.fit(X_users_train, y_users_train)
        user_pred = self.user_classifier.predict(X_users_test)
        user_accuracy = accuracy_score(y_users_test, user_pred)
        print(f"User Classification Accuracy: {user_accuracy:.2f}")
        
        # Split and train module skip classification model
        X_modules_train, X_modules_test, y_modules_train, y_modules_test = train_test_split(
            X_modules, y_modules, test_size=0.2, random_state=42
        )
        
        self.module_skip_classifier.fit(X_modules_train, y_modules_train)
        module_pred = self.module_skip_classifier.predict(X_modules_test)
        module_accuracy = accuracy_score(y_modules_test, module_pred)
        print(f"Module Skip Classification Accuracy: {module_accuracy:.2f}")
    
    def classify_user(self, user_features):
        """
        Classify user based on their performance.
        user_features = [score, time_taken]
        """
        return self.user_classifier.predict([user_features])[0]
    
    def can_skip_module(self, module_features):
        """
        Determine if user can skip the next module.
        module_features = [score, time_taken]
        """
        return bool(self.module_skip_classifier.predict([module_features])[0])

# Initialize the model
model = AdaptiveELearningModel()

@app.route('/categories', methods=['GET'])
def get_categories():
    categories = list(questions_collection.distinct("category"))
    return jsonify([{"id": cat, "name": cat} for cat in categories])

@app.route('/modules/<category>', methods=['GET'])
def get_modules(category):
    modules = list(questions_collection.find(
        {"category": category},
        {"module_id": 1, "module_name": 1, "_id": 0}
    ))
    return jsonify(modules)

@app.route('/questions/<module_id>', methods=['GET'])
def get_questions(module_id):
    module = questions_collection.find_one({"module_id": module_id}, {"questions": 1, "_id": 0})
    if module:
        return jsonify(module['questions'])
    return jsonify([])

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    user_answers = data['user_answers']
    module_id = data['module_id']
    
    module = questions_collection.find_one({"module_id": module_id}, {"questions": 1, "category": 1, "_id": 0})
    questions = module['questions']
    category = module['category']
    
    # Calculate score
    score = sum(1 for user_ans in user_answers 
                for q in questions 
                if q['question_id'] == user_ans['question_id'] and q['correct_option'] == user_ans['answer'])
    
    total_questions = len(questions)
    percentage_score = (score / total_questions) * 100
    
    # Get user features and classify
    time_taken = data.get('time_taken', 30)  # Time taken in minutes
    user_features = [percentage_score, time_taken]  # Score first, time second to match training data
    classification = model.classify_user(user_features)
    
    # Determine if user can skip next module
    module_features = [percentage_score, time_taken]
    can_skip = model.can_skip_module(module_features)
    
    # Get next module information
    all_modules = list(questions_collection.find(
        {"category": category},
        {"module_id": 1, "module_name": 1, "_id": 0}
    ))
    current_module_index = next((i for i, m in enumerate(all_modules) if m['module_id'] == module_id), -1)
    next_module = all_modules[current_module_index + 1] if current_module_index < len(all_modules) - 1 else None
    
    return jsonify({
        'classification': classification,
        'can_skip': can_skip,
        'score': percentage_score,
        'explanation': f"Your score is {percentage_score}% and you have been classified as {classification} level. "
                      f"You {'can' if can_skip else 'cannot'} skip the next module.",
        'next_module': next_module
    })

@app.route('/lessons/<module_id>', methods=['GET'])
def get_lessons(module_id):
    module = questions_collection.find_one({"module_id": module_id}, {"lessons": 1, "_id": 0})
    lessons = module.get("lessons", []) if module else []
    return jsonify(lessons)

if __name__ == '__main__':
    app.run(debug=True)