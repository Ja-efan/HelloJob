package com.ssafy.hellojob.global.auth.service.handler;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Slf4j
@Component
public class OAuth2FailureHandler implements AuthenticationFailureHandler {

    @Override
    public void onAuthenticationFailure(HttpServletRequest request,
                                        HttpServletResponse response,
                                        AuthenticationException exception) throws IOException, ServletException {

        // OAuth 실패 시 상세 로깅
        log.error("OAuth2 failure: {}", exception.getMessage());
        log.info("Original State: {}", request.getSession().getAttribute("OAUTH2_STATE"));
        log.info("Received State: {}", request.getParameter("state"));
        log.info(request.getRequestURI());
        log.info("Code: {}", request.getParameter("code"));
        log.info("Error: {}", request.getParameter("error"));
        log.info("Error Description: {}", request.getParameter("error_description"));

        // 인증 코드 유효성 관련 에러 체크
        if (exception.getMessage().contains("invalid_grant")) {
            log.error("인증 코드가 만료되었거나 이미 사용됨");
        }
        if (request.getParameter("state") == null) {
            log.error("state 파라미터가 누락됨");
        }
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
    }
}