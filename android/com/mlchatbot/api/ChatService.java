// Removed package declaration as file is in default package

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.util.Log;

import com.mlchatbot.ui.MessageItem;
import com.mlchatbot.services.ChatResponseConverter;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * Service to communicate with vAIn_p2p_AGI chat backend
 */
public class ChatService {
    private static final String TAG = "ChatService";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    
    private final OkHttpClient client;
    private final String baseUrl;
    private final ExecutorService executorService;
    private final List<MessageItem> messageHistory;
    private final Context context;

    public ChatService(Context context, String baseUrl) {
        this.context = context;
        this.client = new OkHttpClient();
        this.baseUrl = baseUrl;
        this.executorService = Executors.newSingleThreadExecutor();
        this.messageHistory = new CopyOnWriteArrayList<>();
    }

    public interface ChatCallback {
        void onResponse(MessageItem response);
        void onError(String errorMessage);
    }

    /**
     * Send a message to the chatbot
     * 
     * @param message User's message
     * @param callback Callback to receive the response
     */
    public void sendMessage(String message, ChatCallback callback) {
        // Add user message to history immediately
        MessageItem userMessage = MessageItem.createUserMessage(message);
        messageHistory.add(userMessage);

        // Add a pending response to show waiting status
        if (isNetworkAvailable()) {
            final MessageItem pendingMessage = ChatResponseConverter.createPendingResponse();
            messageHistory.add(pendingMessage);
            
            executorService.execute(() -> {
                try {
                    // Create request body
                    JSONObject requestJson = new JSONObject();
                    requestJson.put("message", message);
                    requestJson.put("user_id", "android_user");
                    
                    RequestBody body = RequestBody.create(
                            requestJson.toString(), 
                            JSON
                    );
                    
                    // Create request
                    Request request = new Request.Builder()
                            .url(baseUrl + "/chat/android")
                            .post(body)
                            .build();
                    
                    // Execute request
                    try (Response response = client.newCall(request).execute()) {
                        if (response.isSuccessful() && response.body() != null) {
                            String responseString = response.body().string();
                            MessageItem botResponse = ChatResponseConverter.fromJson(responseString);
                            
                            // Remove pending message and add actual response
                            messageHistory.remove(pendingMessage);
                            messageHistory.add(botResponse);
                            
                            // Notify callback on main thread
                            callback.onResponse(botResponse);
                            
                        } else {
                            throw new IOException("Unexpected response " + response);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error sending message", e);
                    messageHistory.remove(pendingMessage);
                    callback.onError("Error: " + e.getMessage());
                }
            });
        } else {
            // Handle offline mode
            final MessageItem offlineMessage = ChatResponseConverter.createOfflineResponse();
            messageHistory.add(offlineMessage);
            callback.onResponse(offlineMessage);
        }
    }

    /**
     * Check if network is available
     */
    private boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager = (ConnectivityManager) 
                context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (connectivityManager != null) {
            NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
            return activeNetworkInfo != null && activeNetworkInfo.isConnected();
        }
        return false;
    }

    /**
     * Get message history
     */
    public List<MessageItem> getMessageHistory() {
        return new ArrayList<>(messageHistory);
    }
    
    /**
     * Clear message history
     */
    public void clearHistory() {
        messageHistory.clear();
    }
    
    /**
     * Close resources
     */
    public void shutdown() {
        executorService.shutdown();
    }
}
