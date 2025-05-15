package com.ssafy.hellojob.global.auth.controller;

import com.ssafy.hellojob.global.auth.dto.LoginDto;
import com.ssafy.hellojob.global.auth.service.AuthService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final AuthService authService;

    @GetMapping("/login")
    public LoginDto checkLogin(@CookieValue(value = "access_token") String accessToken, @AuthenticationPrincipal UserPrincipal userPrincipal) {
        return authService.loginStatus(accessToken, userPrincipal.getNickname());
    }

    @ResponseStatus(HttpStatus.NO_CONTENT)
    @PostMapping("/refresh")
    public void refreshToken(HttpServletRequest request, HttpServletResponse response) {
        authService.refreshToken(request, response);
    }

    @ResponseStatus(HttpStatus.NO_CONTENT)
    @PostMapping("/logout")
    public void logout(@AuthenticationPrincipal UserPrincipal user, HttpServletResponse response) {
        authService.logout(user.getUser(), response);
    }


}
