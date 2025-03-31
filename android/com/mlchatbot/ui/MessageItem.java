package com.mlchatbot.ui;

/**
 * Represents a message item in the chat UI.
 */
public class MessageItem {
    private final String message;
    private final boolean isUser;
    private final long timestamp;
    private final String id;
    private final double confidence;

    /**
     * Creates a new message item.
     * 
     * @param message The text content of the message
     * @param isUser Whether this message is from the user (true) or bot (false)
     * @param id Unique identifier for the message
     * @param timestamp The message timestamp
     * @param confidence The confidence score for bot messages (0-1)
     */
    public MessageItem(String message, boolean isUser, String id, long timestamp, double confidence) {
        this.message = message;
        this.isUser = isUser;
        this.id = id;
        this.timestamp = timestamp;
        this.confidence = confidence;
    }

    /**
     * Creates a user message.
     * 
     * @param message The message text
     * @return A new MessageItem for a user message
     */
    public static MessageItem createUserMessage(String message) {
        return new MessageItem(message, true, String.valueOf(System.currentTimeMillis()), 
                              System.currentTimeMillis(), 1.0);
    }

    /**
     * Creates a bot message.
     * 
     * @param message The message text
     * @param id Message ID
     * @param confidence Confidence score
     * @return A new MessageItem for a bot message
     */
    public static MessageItem createBotMessage(String message, String id, double confidence) {
        return new MessageItem(message, false, id, System.currentTimeMillis(), confidence);
    }

    public String getMessage() {
        return message;
    }

    public boolean isUser() {
        return isUser;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getId() {
        return id;
    }

    public double getConfidence() {
        return confidence;
    }
}
