package com.ssafy.hellojob.global.auth.service;

import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.service.UserReadService;
import com.ssafy.hellojob.global.auth.dto.LoginDto;
import com.ssafy.hellojob.global.auth.dto.TokenDto;
import com.ssafy.hellojob.global.auth.entity.Auth;
import com.ssafy.hellojob.global.auth.repository.AuthRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.util.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseCookie;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

import static com.ssafy.hellojob.global.exception.ErrorCode.INVALID_TOKEN;

@Service
@RequiredArgsConstructor
@Transactional
public class AuthService {

    private final JwtUtil jwtUtil;
    private final AuthRepository authRepository;
    private final UserReadService userReadService;

    public void updateRefreshToken(String refreshToken, Auth auth) {
        auth.updateToken(refreshToken, LocalDateTime.now().plusDays(14));
        authRepository.save(auth);
    }

    public void addRefreshTokenCookie(HttpServletResponse response, String refreshToken) {
        ResponseCookie cookie = ResponseCookie.from("refresh_token", refreshToken)
                .httpOnly(true) // JavaScript 접근 차단
                .secure(true)   // HTTPS에서만 사용 가능
                .sameSite("None") // SameSite=None 설정
                .path("/")      // 전체 경로에서 유효
                .maxAge(1209600)  // 14일 유지
                .build();

        addCookieToResponse(cookie, response);
    }

    public void addAccessTokenCookie(HttpServletResponse response, String accessToken) {
        ResponseCookie cookie = ResponseCookie.from("access_token", accessToken)
                .httpOnly(true) // JavaScript 접근 차단
                .secure(true)   // HTTPS에서만 사용 가능
                .sameSite("None") // SameSite=None 설정
                .path("/")      // 전체 경로에서 유효
                .maxAge(32400)  // 9시간 유지
                .build();

        addCookieToResponse(cookie, response);
    }

    private void addCookieToResponse(ResponseCookie cookie, HttpServletResponse response) {
        response.addHeader("Set-Cookie", cookie.toString());
    }


    public LoginDto loginStatus(String accessToken, String nickname) {
        jwtUtil.validateToken(accessToken);
        return new LoginDto(true, nickname);
    }

    public void refreshToken(HttpServletRequest request, HttpServletResponse response) {

        String refreshToken = jwtUtil.extractRefreshToken(request);
        if (!jwtUtil.validateToken(refreshToken)) {
            throw new BaseException(INVALID_TOKEN);
        }

        String email = jwtUtil.getEmailFromToken(refreshToken);
        User user = userReadService.findUserByEmailOrElseThrow(email);
        Auth savedAuth = authRepository.findByUser(user)
                .orElseThrow(() -> new BaseException(INVALID_TOKEN));

        if (!savedAuth.getRefreshToken().equals(refreshToken)) {
            throw new BaseException(INVALID_TOKEN);
        }

        TokenDto newTokens = jwtUtil.generateTokens(email);
        updateRefreshToken(newTokens.getRefreshToken(), savedAuth);
        addRefreshTokenCookie(response, newTokens.getRefreshToken());
        addAccessTokenCookie(response, newTokens.getAccessToken());
    }

    @Transactional
    public void logout(User user, HttpServletResponse response) {
        deleteAuthIfExists(user);
        deleteCookie("access_token", response);
        deleteCookie("refresh_token", response);
    }

    private void deleteAuthIfExists(User user) {
        if (authRepository.existsByUser(user)) {
            authRepository.deleteByUser(user);
        }
    }

    private void deleteCookie(String name, HttpServletResponse response) {
        ResponseCookie cookie = ResponseCookie.from(name, "")
                .httpOnly(true)
                .secure(true)  // HTTPS에서만 사용 가능
                .sameSite("None") // SameSite=None 설정
                .path("/")
                .maxAge(0)
                .build();

        addCookieToResponse(cookie, response);
    }


}
