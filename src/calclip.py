from calorie_clip import CalorieCLIP

# Load model
model = CalorieCLIP.from_pretrained("HaploLLC/CalorieCLIP")

# Predict calories
calories = model.predict("food_photo.jpg")
print(f"Estimated: {calories:.0f} calories")

# Batch prediction
# images = ["breakfast.jpg", "lunch.jpg", "dinner.jpg"]
# results = model.predict_batch(images)
