// chat.js - Script for vAIn Chat room functionality
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const messagesContainer = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const connectionStatus = document.getElementById('connection-status');
    const userCountDisplay = document.getElementById('user-count');
    const offlineIndicator = document.getElementById('offline-indicator');
    
    // Get username from URL or localStorage
    const urlParams = new URLSearchParams(window.location.search);
    let username = urlParams.get('username');
    
    if (!username) {
        username = localStorage.getItem('username');
        if (!username) {
            username = prompt('Please enter your username:');
            if (!username) {
                username = 'Anonymous_' + Math.floor(Math.random() * 1000);
            }
            localStorage.setItem('username', username);
        }
    }
    
    // Initialize Socket.io connection
    const socket = io();
    
    // Queue for messages when offline
    let offlineQueue = [];
    let isOffline = false;
    
    // Socket connection status
    socket.on('connect', function() {
        connectionStatus.textContent = 'Online';
        connectionStatus.className = 'online';
        offlineIndicator.classList.add('hidden');
        isOffline = false;
        
        // Join the room
        socket.emit('join_room', {
            room_id: ROOM_ID,
            username: username
        });
        
        // Send any queued messages
        if (offlineQueue.length > 0) {
            sendQueuedMessages();
        }
    });
    
    socket.on('disconnect', function() {
        connectionStatus.textContent = 'Offline';
        connectionStatus.className = 'offline';
        offlineIndicator.classList.remove('hidden');
        isOffline = true;
    });
    
    socket.on('connection_status', function(data) {
        connectionStatus.textContent = data.status === 'online' ? 'Online' : 'Offline';
        connectionStatus.className = data.status;
        
        if (data.status === 'offline') {
            offlineIndicator.classList.remove('hidden');
            isOffline = true;
        } else {
            offlineIndicator.classList.add('hidden');
            isOffline = false;
            
            // Send any queued messages
            if (offlineQueue.length > 0) {
                sendQueuedMessages();
            }
        }
    });
    
    // Room events
    socket.on('user_joined', function(data) {
        addMessage(data.message);
        userCountDisplay.textContent = `Users: ${data.user_count}`;
    });
    
    socket.on('user_left', function(data) {
        addMessage(data.message);
        userCountDisplay.textContent = `Users: ${data.user_count}`;
    });
    
    socket.on('message_history', function(messages) {
        messagesContainer.innerHTML = '';
        messages.forEach(addMessage);
        scrollToBottom();
    });
    
    socket.on('new_message', function(message) {
        addMessage(message);
        scrollToBottom();
    });
    
    socket.on('message_queued', function(data) {
        // Show the message in the UI with "queued" status
        const tempMessage = data.queued_message;
        tempMessage.queued = true;
        addMessage(tempMessage);
        scrollToBottom();
    });
    
    socket.on('error', function(data) {
        console.error('Socket error:', data.message);
        alert(`Error: ${data.message}`);
    });
    
    // Send message
    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        if (!messageText) return;
        
        if (isOffline) {
            // Queue message for later
            const queuedMessage = {
                content: messageText,
                timestamp: new Date().toISOString()
            };
            offlineQueue.push(queuedMessage);
            
            // Show queued message in UI
            const tempId = 'temp_' + Date.now();
            const tempMessage = {
                id: tempId,
                type: 'user',
                content: messageText,
                sender: username,
                timestamp: new Date().toISOString(),
                queued: true
            };
            
            addMessage(tempMessage);
            scrollToBottom();
        } else {
            // Send message directly
            socket.emit('send_message', {
                message: messageText
            });
        }
        
        messageInput.value = '';
    });
    
    // Helper functions
    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, function(match) {
            const escapeMap = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            return escapeMap[match];
        });
    }

    function addMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.id = message.id;
        
        if (message.type === 'system') {
            messageDiv.classList.add('system');
            messageDiv.innerHTML = escapeHTML(message.content);
        } else {
            // Determine if this is the current user's message
            const isCurrentUser = message.sender === username;
            messageDiv.classList.add(isCurrentUser ? 'user' : 'others');
            
            let queuedLabel = '';
            if (message.queued) {
                queuedLabel = ' <span class="queued-label">(queued)</span>';
            }
            
            messageDiv.innerHTML = `
                <div class="message-sender">${escapeHTML(message.sender)}${queuedLabel}</div>
                <div class="message-content">${escapeHTML(message.content)}</div>
                <div class="message-time">${formatTime(message.timestamp)}</div>
            `;
        }
        
        messagesContainer.appendChild(messageDiv);
    }
    
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    function sendQueuedMessages() {
        console.log(`Sending ${offlineQueue.length} queued messages`);
        
        // Process each queued message
        while (offlineQueue.length > 0) {
            const queuedMessage = offlineQueue.shift();
            socket.emit('send_message', {
                message: queuedMessage.content
            });
        }
    }
    
    // Check server status periodically
    function checkServerStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                // Update connection status
                connectionStatus.textContent = data.status === 'offline' ? 'Offline' : 'Online';
                connectionStatus.className = data.status;
                
                if (data.status === 'offline') {
                    offlineIndicator.classList.remove('hidden');
                    isOffline = true;
                } else {
                    offlineIndicator.classList.add('hidden');
                    isOffline = false;
                    
                    // If we have queued messages, try sending them
                    if (offlineQueue.length > 0 && socket.connected) {
                        sendQueuedMessages();
                    }
                }
            })
            .catch(() => {
                // If can't reach the server, assume offline
                connectionStatus.textContent = 'Offline';
                connectionStatus.className = 'offline';
                offlineIndicator.classList.remove('hidden');
                isOffline = true;
            });
    }
    
    // Check status every 30 seconds
    setInterval(checkServerStatus, 30000);
    
    // Handle leaving the room when navigating away
    window.addEventListener('beforeunload', function() {
        socket.emit('leave_room');
    });
});