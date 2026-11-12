from pipeline import generate_code, render_and_merge

prompt = "Create a 60-second educational video explaining the concept of photosynthesis. The video should be targeted towards high school students and have a friendly, engaging tone. Use a bright color palette with green and yellow as primary colors. Include key concepts such as the role of sunlight, chlorophyll, and the chemical equation for photosynthesis. Ensure that the video is visually appealing with animations of plants, sunlight, and the process of photosynthesis happening in real-time."

scene_codes = generate_code(prompt)
final_video = render_and_merge(scene_codes)
print(final_video)
