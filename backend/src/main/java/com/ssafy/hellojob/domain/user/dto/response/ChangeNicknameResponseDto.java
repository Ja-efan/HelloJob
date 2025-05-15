package com.ssafy.hellojob.domain.user.dto.response;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class ChangeNicknameResponseDto {
    private String nickname;
    private String message;
}
