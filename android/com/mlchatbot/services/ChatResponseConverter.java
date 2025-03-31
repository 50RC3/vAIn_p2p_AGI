package com.mlchatbot.services;

import com.mlchatbot.ui.MessageItem;
import org.json.JSONObject;
import org.json.JSONException;
import java.util.ArrayList;
import java.util.List;

/**
 * Converts between vAIn ChatResponse format and Android MessageItem objects
 */
public class ChatResponseConverter {

    /**
     * Converts a JSON representation of a ChatResponse to a MessageItem
     *
     * @param jsonResponse JSON string containing ChatResponse data
     * @return A MessageItem for displaying in the UI
     * @throws JSONException If the JSON is invalid
     */
    public static MessageItem fromJson(String jsonResponse) throws JSONException {
        JSONObject response = new JSONObject(jsonResponse);
        String id = response.optString("id", String.valueOf(System.currentTimeMillis()));
        String text = response.optString("text", "No response text");
        double confidence = response.optDouble("confidence", 0.0);
        boolean error = response.optBoolean("error", false);

        // If there's an error, format the response differently
        if (error) {
            text = "Error: " + text;
            confidence = 0.0;
        }

        return MessageItem.createBotMessage(text, id, confidence);
    }

    /**
     * Converts a batch of responses to a list of MessageItems
     * 
     * @param jsonBatch JSON array string of responses
     * @return List of MessageItem objects
     * @throws JSONException If JSON parsing fails
     */
    public static List<MessageItem> fromJsonBatch(String jsonBatch) throws JSONException {
        List<MessageItem> items = new ArrayList<>();
        
        // Parse JSON array and convert each item
        // This is a simplified version - actual implementation would handle JSON arrays
        
        return items;
    }

    /**
     * Creates a MessageItem representing a pending response while waiting for server
     * 
     * @return A placeholder MessageItem
     */
    public static MessageItem createPendingResponse() {
        return MessageItem.createBotMessage("Thinking...", "pending", 0.0);
    }
    
    /**
     * Creates a MessageItem for offline response
     * 
     * @return An offline placeholder MessageItem
     */
    public static MessageItem createOfflineResponse() {
        return MessageItem.createBotMessage(
            "I'm currently offline. Your message will be processed when connectivity is restored.", 
            "offline", 
            1.0
        );
    }
}
