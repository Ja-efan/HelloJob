package com.ssafy.hellojob.global.filter;

import com.ssafy.hellojob.global.util.SseUtil;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Slf4j
@Component
public class SseExceptionFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        try {
            filterChain.doFilter(request, response);
        } catch (Exception e) {
            if (SseUtil.isSseRequest(request)) {
                if (!response.isCommitted()) {
                    log.debug("❌ SSE 예외 발생 → SSE 응답 형식으로 처리: {}", e.getMessage());
                    int status = (e instanceof org.springframework.security.access.AccessDeniedException) ?
                            HttpServletResponse.SC_FORBIDDEN : HttpServletResponse.SC_UNAUTHORIZED;
                    SseUtil.writeSseError(response, status, "SSE 인증/인가 실패");
                    return;
                }
            }
        throw e;
    }
}

}
