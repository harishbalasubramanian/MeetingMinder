import re
from transformers import pipeline
from typing import List

def extract_action_items(text: str) -> List[str]:
    patterns = [
        (r'(?i)\b(to\s+(?:develop|create|improve|make sure|build|implement|enhance|finalize|design)\b[\s\w]+?)(?=\s*(?:,|and|\.|$))',
         "direct_actions"),
        (r'(?i)action items?.*?are.*?((?:to\s+[^.]+\n?)+)',
         "action_list")
    ]
    
    action_items = []
    
    for pattern, pattern_type in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if pattern_type == "direct_actions":
                action_items.append(match.group(1))
            elif pattern_type == "action_list":
                full_list = match.group(1)
                items = re.split(r'\s*\bto\s+', full_list)
                action_items.extend([f"to {item.strip('., ')}" for item in items if item.strip()])
    
    return sorted(list(set(action_items)), key=len, reverse=True)[:5]

if __name__ == "__main__":
    test_text = "the action items from here are to develop a front end to improve scalability and the last to make sure to create a strong demo video"
    print(extract_action_items(test_text))

def ml_action_items(text: str) -> List[str]:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["action item", "decision", "information"]
    result = classifier(text, candidate_labels)
    return [result["labels"][0]] if result["scores"][0] > 0.7 else []


# if __name__ == '__main__':
#     test_transcript = "...the action items from here are to develop a front end to improve scalability and the last to just make sure to create a strong demo video..."

#     print(extract_action_items(test_transcript))

