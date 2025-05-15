package com.ssafy.hellojob.global.auth.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class LoginDto {

    private Boolean authenticated;
    private String nickname;

}
