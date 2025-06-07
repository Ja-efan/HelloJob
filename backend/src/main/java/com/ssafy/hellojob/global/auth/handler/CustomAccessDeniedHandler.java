package com.ssafy.hellojob.global.auth.handler;

import com.ssafy.hellojob.global.util.SseUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.web.access.AccessDeniedHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class CustomAccessDeniedHandler implements AccessDeniedHandler {

    @Override
    public void handle(HttpServletRequest request, HttpServletResponse response, AccessDeniedException ex) throws IOException {
        if (!response.isCommitted()) {
            if (SseUtil.isSseRequest(request)) {
                SseUtil.writeSseError(response, HttpServletResponse.SC_FORBIDDEN, "Forbidden (Access Denied)\n\n");
            } else {
                response.sendError(HttpServletResponse.SC_FORBIDDEN, "Access Denied");
            }
        }
    }


}
