package com.ssafy.hellojob.global.util;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;

public class SseUtil {
    public static boolean isSseRequest(HttpServletRequest request) {
        String accept = request.getHeader("Accept");
        return request.getRequestURI().startsWith("/sse") ||
                (accept != null && accept.contains("text/event-stream"));
    }

    public static void writeSseError(HttpServletResponse response, int statusCode, String message) {
        if (response.isCommitted()) return;
        try {
            response.setStatus(statusCode);
            response.setContentType("text/event-stream");
            response.getWriter().write("event: error\ndata: " + message + "\n\n");
            response.getWriter().flush();
        } catch (IOException ignored) {}
    }
}
