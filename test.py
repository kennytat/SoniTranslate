from gradio_client import Client

client = Client("http://localhost:7901/")
result = client.predict(
		text="Tất cả mọi người đều sinh ra có quyền bình đẳng, Tạo hóa cho họ những quyền không ai có thể xâm phạm được; trong những quyền ấy, có quyền được sống, quyền tự do và quyền mưu cầu hạnh phúc.",
		output_file="",
		TRANSLATE_AUDIO_TO="Vietnamese (vi)",
		tts_voice="vn_anh_female",
		speed=1,
		desired_duration="",
		start_time="",
		t2s_method="VietTTS",
		api_name="/tts"
)
print(result)