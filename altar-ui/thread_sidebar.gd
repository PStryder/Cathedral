extends Control

signal thread_switched(new_uid: String)

@onready var chat_log = $/root/Main/HBoxContainer/MainArea/ScrollContainer/ChatLog
@onready var thread_list = $ThreadListScroll/ThreadList
@onready var thread_request = $/root/Main/ThreadListRequest
@onready var activate_request = $/root/Main/ThreadActivateRequest
@onready var new_thread_button = $Button
@onready var chat_scroll = $/root/Main/HBoxContainer/MainArea/ScrollContainer

var active_uid = ""
var active_button: Button = null

func initialize():
	print("⚡ ThreadSidebar ready, loading threads...")
	thread_request.request_completed.connect(_on_thread_list_fetched)
	activate_request.request_completed.connect(_on_thread_activation_response)
	new_thread_button.pressed.connect(_on_create_new_thread_pressed)

	print("Connections complete")
	await get_tree().process_frame
	_load_threads()
	_highlight_active_thread()

func _load_threads():
	var url = "http://127.0.0.1:8000/thread/list/"
	thread_request.request(url)
	print("Threads requested")

func _on_thread_list_fetched(_result, response_code, _headers, body):
	print("🎯 Thread list fetch response received")
	if response_code == 200:
		var json = JSON.new()
		var body_as_string = body.get_string_from_utf8()
		var parse_result = json.parse(body_as_string)
		
		if parse_result == OK:
			print("Threads passed parsing")
			var threads = json.get_data()["threads"]
			print("📦 Raw threads from server: ", threads)
			for child in thread_list.get_children():
				thread_list.remove_child(child)
				child.queue_free()
			print("Children after clear: ", thread_list.get_child_count())
			
			for thread in threads:
				var button = Button.new()
				button.set_meta("thread_uid", thread["thread_uid"])
				button.text = thread.get("thread_name", thread["thread_uid"].substr(0, 8))
				button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
				button.size_flags_vertical = Control.SIZE_FILL
				button.custom_minimum_size = Vector2(0, 40)
				button.clip_text = false
				button.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
				#button.align = HORIZONTAL_ALIGNMENT_LEFT
				#button.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
				button.focus_mode = Control.FOCUS_NONE
				button.pressed.connect(Callable(self, "_on_thread_selected").bind(thread["thread_uid"]))
				thread_list.add_child(button)
			# 👇 Auto-select first thread
			if threads.size() > 0:
				var first_thread = threads[0]
				active_uid = first_thread["thread_uid"]
				_on_thread_selected(active_uid)
				add_message_to_chat("[i]✔ Threads loaded.[/i]")
		else:
			print("❌ Error parsing thread list JSON.")
	else:
		print("❌ Failed to fetch thread list. Code: " + str(response_code))

func _on_thread_selected(thread_uid: String):
	for child in thread_list.get_children():
		if child is Button:
			if child.get_meta("thread_uid", "") == thread_uid:
				active_button = child
				child.add_theme_color_override("font_color", Color.html("f9e46d"))  # golden
				var style = StyleBoxFlat.new()
				style.bg_color = Color.html("303020")
				style.border_color = Color.html("f9e46d")
				style.border_width_left = 3
				child.add_theme_stylebox_override("normal", style)
				if not child.text.begins_with("✴ "):
					child.text = "✴ " + child.text.strip_edges()
			else:
				child.add_theme_color_override("font_color", Color.html("ffffff"))
				var style = StyleBoxFlat.new()
				style.bg_color = Color.html("202020")
				style.border_width_left = 0
				child.add_theme_stylebox_override("normal", style)
				child.text = child.text.replace("✴ ", "")

	active_uid = thread_uid
	var url = "http://127.0.0.1:8000/thread/"
	var body = {"thread_uid": thread_uid}
	var json_body = JSON.stringify(body)
	var headers = ["Content-Type: application/json"]
	activate_request.request(url, headers, HTTPClient.METHOD_POST, json_body)
	_highlight_active_thread()

func _on_thread_activation_response(_result, response_code, _headers, body):
	if response_code == 200:
		var json = JSON.new()
		var body_as_string = body.get_string_from_utf8()
		var parse_result = json.parse(body_as_string)

		if parse_result == OK:
			active_uid = json.get_data()["thread_uid"]
			add_message_to_chat("[left]🧵 Activated thread: [b]" + active_uid + "[/b][/left]\n")
			await get_tree().process_frame
			chat_scroll.scroll_vertical = chat_scroll.get_v_scroll_bar().max_value
			thread_switched.emit(active_uid)
			_highlight_active_thread()
		else:
			add_message_to_chat("[left]❌ Error activating thread.[/left]\n")
	else:
		add_message_to_chat("[left]❌ Thread activation failed: " + str(response_code) + "[/left]\n")

func _on_create_new_thread_pressed():
	var url = "http://127.0.0.1:8000/thread/new/"
	var headers = ["Content-Type: application/json"]
	var json_body = JSON.stringify({ "thread_name": "New Thread" })
	var create_request = get_node("/root/Main/ThreadCreateRequest")

	if not create_request.is_connected("request_completed", Callable(self, "_on_new_thread_created")):
		create_request.request_completed.connect(Callable(self, "_on_new_thread_created"))

	create_request.request(url, headers, HTTPClient.METHOD_POST, json_body)

func _on_new_thread_created(_result, response_code, _headers, body):
	if response_code == 200:
		var json = JSON.new()
		var parse_result = json.parse(body.get_string_from_utf8())
		if parse_result == OK:
			var thread = json.get_data()
			active_uid = thread["thread_uid"]
			_load_threads()
			_on_thread_selected(thread["thread_uid"])

func _highlight_active_thread():
	for child in thread_list.get_children():
		if child is Button:
			var uid = child.get_meta("thread_uid", "")
			var style = StyleBoxFlat.new()
			
			if uid == active_uid:
				child.add_theme_color_override("font_color", Color.html("#f9e46d"))  # gold
				style.bg_color = Color.html("#3a3a20")
				style.border_color = Color.html("#f9e46d")
				style.border_width_left = 4
				child.add_theme_stylebox_override("normal", style)

				if not child.text.begins_with("✴ "):
					child.text = "✴ " + child.text.strip_edges()
			else:
				child.add_theme_color_override("font_color", Color.html("#ffffff"))  # default
				style.bg_color = Color.html("#202020")
				style.border_width_left = 0
				child.add_theme_stylebox_override("normal", style)
				child.text = child.text.replace("✴ ", "")

func add_message_to_chat(text: String, role := "assistant"):
	var row = HBoxContainer.new()
	row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	row.alignment = BoxContainer.ALIGNMENT_BEGIN

	var filler_left = Control.new()
	filler_left.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var filler_right = Control.new()
	filler_right.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var bubble = Panel.new()
	bubble.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	bubble.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	bubble.custom_minimum_size = Vector2(0, 0)
	bubble.add_theme_color_override("panel_color", Color(0, 0, 0, 1)) # solid black
	bubble.add_theme_constant_override("margin_left", 8)
	bubble.add_theme_constant_override("margin_right", 8)
	bubble.add_theme_constant_override("margin_top", 4)
	bubble.add_theme_constant_override("margin_bottom", 4)

	var label = RichTextLabel.new()
	label.bbcode_enabled = true
	label.bbcode_text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.fit_content = true
	label.scroll_active = false
	label.size_flags_horizontal = Control.SIZE_FILL
	label.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	label.custom_minimum_size.x = 600

	bubble.add_child(label)

	if role == "assistant":
		filler_left.visible = false
	else:
		filler_right.visible = false

	row.add_child(filler_left)
	row.add_child(bubble)
	row.add_child(filler_right)

	chat_log.add_child(row)
	await get_tree().process_frame
	chat_scroll.scroll_vertical = chat_scroll.get_v_scroll_bar().max_value
