import json

categories = [
    "Power_Saving_Tips", "LED_vs_CFL", "AC_Maintenance_Tips",
    "Refrigerator_Efficiency", "Smart_Lighting", "Insulation_Tips", "Appliance_Use"
]

intents = []

for category in categories:
    for i in range(125):  # 125 tags per category to make 1000 in total
        tag = f"{category}_{i+1}"
        patterns = [
            f"Tell me about {category.lower()} {i+1}",
            f"How can I improve {category.lower()}",
            f"Advice on {category.lower()} {i+1}"
        ]
        responses = [
            f"Here's some information about {category.lower()} tag {i+1}.",
            f"Consider these tips for {category.lower()} {i+1}."
        ]
        intents.append({
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        })

with open('intents.json', 'w') as file:
    json.dump({"intents": intents}, file, indent=4)
