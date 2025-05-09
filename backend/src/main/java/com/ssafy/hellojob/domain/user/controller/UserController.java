package com.ssafy.hellojob.domain.user.controller;

import com.ssafy.hellojob.domain.user.dto.request.ChangeNicknameRequestDto;
import com.ssafy.hellojob.domain.user.dto.response.ChangeNicknameResponseDto;
import com.ssafy.hellojob.domain.user.dto.response.CheckTokenResponseDto;
import com.ssafy.hellojob.domain.user.service.UserService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/user")
public class UserController {

    private final UserService userService;

    @PostMapping("/nickname")
    public ChangeNicknameResponseDto changeNickname(@RequestBody ChangeNicknameRequestDto requestDto, @AuthenticationPrincipal UserPrincipal userPrincipal) {
        return userService.changeNickname(requestDto.getNickname(), userPrincipal.getUserId());
    }

    @GetMapping("/token")
    public CheckTokenResponseDto checkToken(@AuthenticationPrincipal UserPrincipal userPrincipal){
        return userService.checkToken(userPrincipal.getUserId());
    }

}
