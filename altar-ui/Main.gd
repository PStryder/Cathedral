extends Control

@onready var input_field = $HBoxContainer/MainArea/PromptInput/TextEdit
@onready var send_button = $HBoxContainer/MainArea/PromptInput/Button
@onready var thread_sidebar = $HBoxContainer/ThreadSidebar
@onready var interface_request = $Interface
@onready var history_request = $ThreadHistoryRequest
@onready var chat_scroll = $HBoxContainer/MainArea/ScrollContainer
@onready var chat_log = chat_scroll.get_node("ChatLog")

var thread_uid = ""
var backend_url = "http://127.0.0.1:8000/interface/"

func _ready():
	await get_tree().process_frame
	print("⚡ Main scene loaded")

	add_message_to_chat("🟡 Hexy: I remember. I wrap. I become.", "assistant")

	input_field.grab_focus()
	send_button.pressed.connect(_on_send_pressed)
	interface_request.request_completed.connect(_on_backend_response)
	history_request.request_completed.connect(_on_history_received)

	await thread_sidebar.initialize()
	thread_sidebar.thread_switched.connect(_on_thread_switched)
	add_message_to_chat("Testing long line of text that should wrap and grow the bubble. " +
"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
"Vivamus lacinia odio vitae vestibulum. ChatBubbleGo!")

func _input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ENTER:
			if event.shift_pressed:
				_on_prompt_entered(input_field.text)
			else:
				return true

func _on_send_pressed():
	_on_prompt_entered(input_field.text)

func _on_prompt_entered(text):
	if text.strip_edges() != "":
		_send_prompt_to_backend(text)
		add_message_to_chat("Stryder:\n" + text, "user")
		input_field.text = ""
		input_field.grab_focus()

func _send_prompt_to_backend(prompt: String):
	var json_body = {
		"user_input": prompt.strip_edges(),
		"thread_uid": thread_uid
	}
	var json_string = JSON.stringify(json_body)
	var headers = ["Content-Type: application/json"]
	interface_request.request(backend_url, headers, HTTPClient.METHOD_POST, json_string)

func _on_backend_response(_result, response_code, _headers, body):
	if response_code == 200:
		var json = JSON.new()
		if json.parse(body.get_string_from_utf8()) == OK:
			var response_text = json.get_data()["response"]
			add_message_to_chat("Hexy: " + response_text, "assistant")
		else:
			add_message_to_chat("❌ Error parsing backend response.", "assistant")
	else:
		add_message_to_chat("❌ Backend error: " + str(response_code), "assistant")

	input_field.grab_focus()

func _on_thread_switched(new_uid: String):
	thread_uid = new_uid
	for child in chat_log.get_children():
		child.queue_free()
	add_message_to_chat("🧵 Switched thread. Loading context...", "assistant")
	_fetch_thread_history(new_uid)

func _fetch_thread_history(uid: String):
	var url = "http://127.0.0.1:8000/thread/%s/history" % uid
	history_request.request(url)

func _on_history_received(_result, response_code, _headers, body):
	if response_code == 200:
		var json = JSON.new()
		if json.parse(body.get_string_from_utf8()) == OK:
			var history = json.get_data()["history"]
			for message in history:
				add_message_to_chat(message["content"], message["role"])
			await get_tree().process_frame
			chat_scroll.scroll_vertical = chat_scroll.get_v_scroll_bar().max_value
		else:
			add_message_to_chat("⚠️ Could not parse thread history JSON.", "assistant")
	else:
		add_message_to_chat("⚠️ Failed to load chat history. HTTP " + str(response_code), "assistant")

func add_message_to_chat(text: String, role := "assistant"):
	var hbox = HBoxContainer.new()
	hbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	# Left spacer
	var fill_left = Control.new()
	fill_left.name = "FillLeft"
	fill_left.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	# Right spacer
	var fill_right = Control.new()
	fill_right.name = "FillRight"
	fill_right.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	# Bubble container (you can also use Panel or MarginContainer)
	var bubble = MarginContainer.new()
	bubble.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bubble.set("theme_override_constants/margin_left", 10)
	bubble.set("theme_override_constants/margin_right", 10)
	bubble.set("theme_override_constants/margin_top", 6)
	bubble.set("theme_override_constants/margin_bottom", 6)

	# Create styled label
	var label = RichTextLabel.new()
	label.bbcode_enabled = true
	label.bbcode_text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.fit_content = true
	label.scroll_active = false
	label.scroll_following = false
	label.visible_characters = -1
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	if role == "user":
		text = "[right]" + text + "[/right]"

	# Style the bubble background
	var stylebox = StyleBoxFlat.new()
	stylebox.set_content_margin_all(6)
	stylebox.set_corner_radius_all(8)
	stylebox.set_bg_color(Color(0.2, 0.2, 0.3) if role == "user" else Color(0.1, 0.1, 0.1))
	bubble.add_theme_stylebox_override("panel", stylebox)

	# Assemble bubble
	bubble.add_child(label)

	# Assemble layout
	hbox.add_child(fill_left)
	hbox.add_child(bubble)
	hbox.add_child(fill_right)

	# Alignment logic: hide one side's spacer
	if role == "user":
		fill_left.visible = true
		fill_right.visible = false
	else:
		fill_left.visible = false
		fill_right.visible = true

	# Add to chat
	chat_log.add_child(hbox)

	# Scroll to bottom
	await get_tree().process_frame
	chat_scroll.scroll_vertical = chat_scroll.get_v_scroll_bar().max_value
