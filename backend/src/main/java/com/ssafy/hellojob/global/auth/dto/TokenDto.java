package com.ssafy.hellojob.global.auth.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class TokenDto {

    private String accessToken;
    private String refreshToken;
    private long accessTokenExpiresIn;
}