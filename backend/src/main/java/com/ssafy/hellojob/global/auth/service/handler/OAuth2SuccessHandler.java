package com.ssafy.hellojob.global.auth.service.handler;

import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.service.UserReadService;
import com.ssafy.hellojob.global.auth.dto.TokenDto;
import com.ssafy.hellojob.global.auth.entity.Auth;
import com.ssafy.hellojob.global.auth.repository.AuthRepository;
import com.ssafy.hellojob.global.auth.service.AuthService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.util.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import java.io.IOException;

import static com.ssafy.hellojob.global.exception.ErrorCode.AUTH_NOT_FOUND;

@Slf4j
@Component
@RequiredArgsConstructor
public class OAuth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    private final JwtUtil jwtUtil;
    private final AuthService authService;
    private final AuthRepository authRepository;
    private final UserReadService userReadService;

    @Value("${frontend.url}")
    private String frontendUrl;

    @Override
    public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
                                        Authentication authentication) throws IOException {
        UserPrincipal oAuth2User = (UserPrincipal) authentication.getPrincipal();
        String email = oAuth2User.getEmail();
        User user = userReadService.findUserByEmailOrElseThrow(email);
        Auth auth = authRepository.findByUser(user)
                .orElseThrow(() -> new BaseException(AUTH_NOT_FOUND));

        // 토큰 생성
        TokenDto tokens = jwtUtil.generateTokens(email);

        // Refresh Token DB 저장
        authService.updateRefreshToken(tokens.getRefreshToken(), auth);

        // Refresh Token 쿠키에 저장
        authService.addRefreshTokenCookie(response, tokens.getRefreshToken());

        // Access Token 쿠키에 저장
        authService.addAccessTokenCookie(response, tokens.getAccessToken());

        // CORS 헤더 설정
        response.setHeader("Access-Control-Allow-Origin", frontendUrl);
        response.setHeader("Access-Control-Allow-Credentials", "true");

        response.sendRedirect(frontendUrl + "/");
    }
}
