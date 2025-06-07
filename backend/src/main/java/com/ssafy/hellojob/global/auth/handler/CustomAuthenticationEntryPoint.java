package com.ssafy.hellojob.global.auth.handler;

import com.ssafy.hellojob.global.util.SseUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
public class CustomAuthenticationEntryPoint implements AuthenticationEntryPoint {

    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response, AuthenticationException ex) throws IOException {
        if (!response.isCommitted()) {
            if (SseUtil.isSseRequest(request)) {
                SseUtil.writeSseError(response, HttpServletResponse.SC_UNAUTHORIZED, "Unauthorized\n\n");
            }
        } else {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Unauthorized");
        }
    }

}
