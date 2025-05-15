package com.ssafy.hellojob.global.exception;

import lombok.Getter;

@Getter
public class BaseException extends RuntimeException{
    private final ErrorCode errorCode;

    public BaseException(ErrorCode errorCode) {
        super(errorCode.getMessage()); // RuntimeException의 message 필드에 ErrorCode의 message를 전달
        this.errorCode = errorCode;
    }
}

