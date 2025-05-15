package com.ssafy.hellojob.domain.user.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class ChangeNicknameRequestDto {

    @NotBlank(message = ValidationMessage.NICKNAME_NOT_EMPTY)
    @Size(min = 1, max = 30, message = ValidationMessage.ERROR_NICKNAME_LENGTH)
    @Pattern(regexp = "^[가-힣a-zA-Z]{1,30}$", message = ValidationMessage.ERROR_NICKNAME_FORMAT)
    private String nickname;

}
