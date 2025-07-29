# Query Intent Classifier

A deep learning solution for classifying user queries as Gmail/Email-related or Calendar-related using modern NLP techniques. This binary classification system helps route natural language queries to the appropriate service in productivity applications.

## 🎯 Project Overview

This project implements a transformer-based classifier that distinguishes between:
- **Gmail queries (0)**: Email-related requests like searching messages, finding attachments, managing inbox
- **Calendar queries (1)**: Schedule-related requests like checking appointments, finding meetings, managing events

## 📊 Model Performance

- **Test Accuracy**: 90.48%
- **Validation Accuracy**: 95.24% 
- **Architecture**: DistilBERT-based sequence classification
- **Training**: 5 epochs with 96 training samples

### Performance Metrics
| Metric | Gmail | Calendar | Overall |
|--------|-------|----------|---------|
| Precision | 0.91 | 0.90 | 0.90 |
| Recall | 0.91 | 0.90 | 0.90 |
| F1-Score | 0.91 | 0.90 | 0.90 |

## 🏗️ Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Classification Head**: Binary sequence classification
- **Max Sequence Length**: 128 tokens
- **Framework**: Hugging Face Transformers with PyTorch

## 📁 Dataset

The dataset contains 138 carefully curated examples:
- **Gmail queries**: 70 examples (email search, attachments, inbox management, etc.)
- **Calendar queries**: 68 examples (meetings, appointments, schedule management, etc.)
- **Split**: 70% train, 15% validation, 15% test
- **Edge cases included**: Ambiguous queries that could apply to both categories

### Example Queries

**Gmail Examples:**
- "Find emails with PDF attachments"
- "Show me unread messages in my inbox" 
- "Search for emails from Sarah about the project"
- "What did I discuss yesterday?"

**Calendar Examples:**
- "When is my next meeting with the design team?"
- "Show me all events scheduled for Tuesday"
- "Find appointments with Dr. Johnson"
- "What's on my schedule for today?"

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
```

### Training the Model

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd query-intent-classifier
   ```

2. **Run the notebook**:
   - Open `query_intent_classifier.ipynb`
   - Execute all cells to train and evaluate the model

3. **Key training parameters**:
   - Learning rate: 2e-5
   - Batch size: 8
   - Epochs: 5
   - Warmup steps: 10

### Making Predictions

```python
def classify_query(text, return_probabilities=False):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        confidence = probabilities[predicted_class_id]

    predicted_class = target_names[predicted_class_id]

    if return_probabilities:
        return predicted_class, predicted_class_id, confidence, probabilities
    return predicted_class, predicted_class_id, confidence

# Example usage
result = classify_query("Find my presentation for next week")
print(f"Classification: {result[0]}, Confidence: {result[2]:.4f}")
```

## 📈 Training Results

The model shows excellent convergence:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 | Precision | Recall |
|-------|---------------|----------------|----------|----|-----------||
| 1 | 0.695 | 0.668 | 0.619 | 0.543 | 0.779 | 0.619 |
| 2 | 0.647 | 0.532 | 0.905 | 0.905 | 0.905 | 0.905 |
| 3 | 0.532 | 0.355 | 0.952 | 0.952 | 0.956 | 0.952 |
| 4 | 0.378 | 0.262 | 0.952 | 0.952 | 0.956 | 0.952 |
| 5 | 0.198 | 0.236 | 0.952 | 0.952 | 0.956 | 0.952 |

## 🔍 Error Analysis

The model misclassifies only 2 out of 21 test examples:
- **False Positive**: "Where's my latest message?" → Predicted Calendar instead of Gmail
- **False Negative**: "Display appointments tagged as urgent" → Predicted Gmail instead of Calendar

These edge cases represent genuinely ambiguous queries that could reasonably be interpreted either way.

## 🛠️ Technical Implementation

### Key Features:
- **Efficient Architecture**: DistilBERT for faster inference while maintaining accuracy
- **Robust Training**: Stratified splits ensure balanced representation
- **Comprehensive Evaluation**: Includes confusion matrix and per-class metrics
- **Production Ready**: Includes confidence scores and probability distributions

### Model Components:
```python
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
```

## 📝 Use Cases

This classifier can be integrated into:
- **Virtual Assistants**: Route user queries to appropriate handlers
- **Productivity Apps**: Automatically categorize user intents
- **Email Clients**: Distinguish between email and calendar requests
- **Voice Interfaces**: Process spoken commands for office automation
- **Chatbots**: Understand user intent for better responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built using Hugging Face Transformers
- DistilBERT model by Hugging Face
- Dataset carefully curated for productivity use cases

---

**Note**: This is a research/educational project. For production use, consider expanding the dataset and implementing additional validation measures.
