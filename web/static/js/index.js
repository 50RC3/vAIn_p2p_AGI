// index.js - Main script for vAIn Chat home page
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const roomsList = document.getElementById('rooms-list');
    const roomNameInput = document.getElementById('room-name');
    const usernameInput = document.getElementById('username');
    const createRoomBtn = document.getElementById('create-room-btn');
    const connectionStatus = document.getElementById('connection-status');
    
    // Initialize Socket.io connection
    const socket = io();
    
    // Load room list on page load
    loadRooms();
    
    // Socket connection status
    socket.on('connect', function() {
        connectionStatus.textContent = 'Online';
        connectionStatus.className = 'online';
    });
    
    socket.on('disconnect', function() {
        connectionStatus.textContent = 'Offline';
        connectionStatus.className = 'offline';
    });
    
    socket.on('connection_status', function(data) {
        connectionStatus.textContent = data.status === 'online' ? 'Online' : 'Offline';
        connectionStatus.className = data.status;
    });
    
    // Create room event
    createRoomBtn.addEventListener('click', function() {
        const roomName = roomNameInput.value.trim();
        const username = usernameInput.value.trim();
        
        if (!roomName) {
            alert('Please enter a room name');
            return;
        }
        
        if (!username) {
            alert('Please enter your username');
            return;
        }
        
        // Save username in local storage for persistence
        localStorage.setItem('username', username);
        
        // Create room via API
        fetch('/api/rooms', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: roomName })
        })
        .then(response => response.json())
        .then(room => {
            // Navigate to the new room
            window.location.href = `/chat/${room.id}?username=${encodeURIComponent(username)}`;
        })
        .catch(error => {
            console.error('Error creating room:', error);
            alert('Failed to create room. Please try again.');
        });
    });
    
    // Load rooms function
    function loadRooms() {
        fetch('/api/rooms')
            .then(response => response.json())
            .then(rooms => {
                if (rooms.length === 0) {
                    roomsList.innerHTML = '<p>No rooms available. Create one below!</p>';
                    return;
                }
                
                // Sort rooms by creation date (newest first)
                rooms.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                
                // Create room list
                let html = '';
                rooms.forEach(room => {
                    html += `
                        <div class="room-item">
                            <div class="room-info">
                                <div class="room-name">${room.name}</div>
                                <div class="room-users">${room.user_count} users online</div>
                            </div>
                            <button class="join-btn" data-room-id="${room.id}">Join</button>
                        </div>
                    `;
                });
                
                roomsList.innerHTML = html;
                
                // Add event listeners to join buttons
                document.querySelectorAll('.join-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const roomId = this.getAttribute('data-room-id');
                        let username = usernameInput.value.trim();
                        
                        // If no username entered, check local storage
                        if (!username) {
                            username = localStorage.getItem('username') || '';
                        }
                        
                        // If still no username, prompt
                        if (!username) {
                            username = prompt('Please enter your username to join the room:');
                            if (!username) return;
                        }
                        
                        // Save username
                        localStorage.setItem('username', username);
                        
                        // Navigate to room
                        window.location.href = `/chat/${roomId}?username=${encodeURIComponent(username)}`;
                    });
                });
            })
            .catch(error => {
                console.error('Error loading rooms:', error);
                roomsList.innerHTML = '<p>Error loading rooms. Please refresh the page.</p>';
            });
    }
    
    // Check server status periodically
    function checkServerStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                // Update connection status
                connectionStatus.textContent = data.status === 'offline' ? 'Offline' : 'Online';
                connectionStatus.className = data.status;
            })
            .catch(() => {
                // If can't reach the server, assume offline
                connectionStatus.textContent = 'Offline';
                connectionStatus.className = 'offline';
            });
    }
    
    // Check status every 30 seconds
    setInterval(checkServerStatus, 30000);
    
    // Prefill username from localStorage if available
    const savedUsername = localStorage.getItem('username');
    if (savedUsername) {
        usernameInput.value = savedUsername;
    }
});