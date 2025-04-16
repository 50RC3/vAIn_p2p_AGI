from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import uuid
import json
import logging
from datetime import datetime
import sys

# Add project root to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules for integration
try:
    from core.interactive_utils import InteractiveSession
    from config.system_config import load_config
    from utils.dependency_checker import check_dependencies
except ImportError as e:
    logging.warning(f"Could not import some vAIn modules: {e}")
    # Continue anyway, as the web server can function independently

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'web_server.log'), 'a')
    ]
)

logger = logging.getLogger('vain_web_server')

# Load configuration if available
config = {}
try:
    config = load_config()
    logger.info("Loaded system configuration")
except Exception as e:
    logger.warning(f"Failed to load system configuration: {e}")
    logger.info("Using default configuration")

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', config.get('web', {}).get('secret_key', 'default_secret_key'))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent', logger=True, engineio_logger=True)

# In-memory storage for rooms and messages (replace with a database in production)
rooms = {}
messages = {}

# Offline operation support (queues messages when offline)
offline_mode = False
message_queue = []
sync_in_progress = False

# Directory for offline storage
offline_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'offline_messages')
os.makedirs(offline_dir, exist_ok=True)

def save_offline_message(message, room_id):
    """Save message for later synchronization when back online"""
    if not offline_mode:
        return
        
    try:
        message_queue.append({'message': message, 'room_id': room_id, 'timestamp': datetime.now().isoformat()})
        # Save to disk to persist messages even if server restarts
        offline_file = os.path.join(offline_dir, f'offline_queue_{datetime.now().strftime("%Y%m%d")}.json')
        
        with open(offline_file, 'w') as f:
            json.dump(message_queue, f)
            
        logger.info(f"Message saved to offline queue ({len(message_queue)} messages pending)")
    except Exception as e:
        logger.error(f"Failed to save offline message: {e}")

def process_offline_queue():
    """Process queued messages once back online"""
    global sync_in_progress, message_queue
    
    if offline_mode or sync_in_progress or not message_queue:
        return
        
    sync_in_progress = True
    logger.info(f"Processing offline message queue ({len(message_queue)} messages)")
    
    try:
        # Sort by timestamp
        message_queue.sort(key=lambda x: x['timestamp'])
        
        # Process messages
        while message_queue:
            item = message_queue.pop(0)
            room_id = item['room_id']
            message = item['message']
            
            if room_id in rooms:
                messages[room_id].append(message)
                socketio.emit('new_message', message, to=room_id)
                logger.debug(f"Processed offline message: {message['id']}")
            
        # Clear saved offline files
        for file in os.listdir(offline_dir):
            if file.startswith('offline_queue_'):
                os.remove(os.path.join(offline_dir, file))
                
        logger.info("Offline queue processing completed")
    except Exception as e:
        logger.error(f"Error processing offline queue: {e}")
    finally:
        sync_in_progress = False

def load_offline_queue():
    """Load any saved offline messages from disk"""
    global message_queue
    
    try:
        for file in os.listdir(offline_dir):
            if file.startswith('offline_queue_'):
                with open(os.path.join(offline_dir, file), 'r') as f:
                    saved_queue = json.load(f)
                    if saved_queue:
                        message_queue.extend(saved_queue)
                        
        if message_queue:
            logger.info(f"Loaded {len(message_queue)} offline messages from disk")
    except Exception as e:
        logger.error(f"Failed to load offline queue: {e}")

# Load any saved offline messages on startup
load_offline_queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat/<room_id>')
def chat_room(room_id):
    if room_id not in rooms:
        return "Room not found", 404
    return render_template('chat.html', room_id=room_id, room_name=rooms[room_id]['name'])

@app.route('/api/rooms', methods=['GET'])
def get_rooms():
    return jsonify(list(rooms.values()))

@app.route('/api/rooms', methods=['POST'])
def create_room():
    data = request.json
    room_id = str(uuid.uuid4())
    rooms[room_id] = {
        'id': room_id,
        'name': data.get('name', 'Unnamed Room'),
        'created_at': datetime.now().isoformat(),
        'user_count': 0
    }
    messages[room_id] = []
    return jsonify(rooms[room_id]), 201

@app.route('/api/rooms/<room_id>/messages', methods=['GET'])
def get_messages(room_id):
    if room_id not in rooms:
        return jsonify({"error": "Room not found"}), 404
    return jsonify(messages.get(room_id, []))

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['room_id']
    username = data.get('username', 'Anonymous')
    
    if room_id not in rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    join_room(room_id)
    session['room_id'] = room_id
    session['username'] = username
    
    rooms[room_id]['user_count'] += 1
    
    # Notify others that user joined
    join_message = {
        'id': str(uuid.uuid4()),
        'type': 'system',
        'content': f'{username} has joined the room',
        'sender': 'System',
        'timestamp': datetime.now().isoformat()
    }
    
    messages[room_id].append(join_message)
    emit('user_joined', {
        'username': username,
        'user_count': rooms[room_id]['user_count'],
        'message': join_message
    }, to=room_id)
    
    # Send existing messages to the newly joined user
    emit('message_history', messages[room_id])

@socketio.on('leave_room')
def handle_leave_room(data):
    room_id = session.get('room_id')
    username = session.get('username', 'Anonymous')
    
    if not room_id or room_id not in rooms:
        return
    
    leave_room(room_id)
    rooms[room_id]['user_count'] = max(0, rooms[room_id]['user_count'] - 1)
    
    leave_message = {
        'id': str(uuid.uuid4()),
        'type': 'system',
        'content': f'{username} has left the room',
        'sender': 'System',
        'timestamp': datetime.now().isoformat()
    }
    
    messages[room_id].append(leave_message)
    emit('user_left', {
        'username': username,
        'user_count': rooms[room_id]['user_count'],
        'message': leave_message
    }, to=room_id)
    
    session.pop('room_id', None)
    session.pop('username', None)

@socketio.on('send_message')
def handle_send_message(data):
    room_id = session.get('room_id')
    username = session.get('username', 'Anonymous')
    
    if not room_id or room_id not in rooms:
        emit('error', {'message': 'Room not found or you are not in a room'})
        return
    
    message = {
        'id': str(uuid.uuid4()),
        'type': 'user',
        'content': data['message'],
        'sender': username,
        'timestamp': datetime.now().isoformat()
    }
    
    if offline_mode:
        save_offline_message(message, room_id)
        emit('message_queued', {
            'message': 'Message queued for delivery when back online',
            'queued_message': message
        })
    else:
        messages[room_id].append(message)
        emit('new_message', message, to=room_id)

# Network connectivity and offline mode management
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status including offline mode"""
    try:
        # Try to import system modules for better integration
        from monitoring.metrics_collector import get_system_metrics
        metrics = get_system_metrics()
    except ImportError:
        metrics = {
            'cpu': 0,
            'memory': 0,
            'uptime': 0
        }
        
    return jsonify({
        'status': 'offline' if offline_mode else 'online',
        'offline_queue_size': len(message_queue),
        'rooms_count': len(rooms),
        'system_metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/offline-mode', methods=['POST'])
def toggle_offline_mode():
    """Enable or disable offline mode"""
    global offline_mode
    
    data = request.json
    new_state = data.get('enabled')
    
    if new_state is None:
        return jsonify({"error": "Missing 'enabled' parameter"}), 400
        
    previous_state = offline_mode
    offline_mode = bool(new_state)
    
    # If coming back online, process the offline queue
    if previous_state and not offline_mode:
        socketio.start_background_task(process_offline_queue)
    
    logger.info(f"Offline mode {'enabled' if offline_mode else 'disabled'}")
    return jsonify({
        'offline_mode': offline_mode,
        'queue_size': len(message_queue)
    })

# Health check endpoint for monitoring systems
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'version': '0.2.1',
        'offline_mode': offline_mode,
        'rooms': len(rooms),
        'uptime': get_uptime(),
        'timestamp': datetime.now().isoformat()
    })

def get_uptime():
    """Get server uptime in seconds"""
    try:
        with open('/proc/uptime', 'r') as f:
            return float(f.readline().split()[0])
    except:
        return 0

# Background task for network connectivity monitoring
def check_network_connectivity():
    """Background task to monitor network connectivity"""
    import time
    import socket
    
    while True:
        # Check if we can connect to a reliable host
        connected = False
        try:
            # Try connecting to Google's DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            connected = True
        except OSError:
            connected = False
            
        # Update offline mode if it doesn't match detected state
        global offline_mode
        if not connected and not offline_mode:
            logger.warning("Network connection lost, enabling offline mode")
            offline_mode = True
            socketio.emit('connection_status', {'status': 'offline'})
        elif connected and offline_mode:
            logger.info("Network connection restored, disabling offline mode")
            offline_mode = False
            socketio.emit('connection_status', {'status': 'online'})
            socketio.start_background_task(process_offline_queue)
            
        time.sleep(30)  # Check every 30 seconds

# Start network connectivity monitoring in background
if config.get('web', {}).get('auto_detect_connectivity', True):
    socketio.start_background_task(check_network_connectivity)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting vAIn web server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)