package com.ssafy.hellojob.global.exception;

import org.springframework.http.HttpStatus;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum ErrorCode {

    // Global Exception
    BAD_REQUEST_ERROR(HttpStatus.BAD_REQUEST, "잘못된 요청입니다."),
    INVALID_HTTP_MESSAGE_BODY(HttpStatus.BAD_REQUEST,"HTTP 요청 바디의 형식이 잘못되었습니다."),
    UNSUPPORTED_HTTP_METHOD(HttpStatus.METHOD_NOT_ALLOWED,"지원하지 않는 HTTP 메서드입니다."),
    SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR,"서버 내부에서 알 수 없는 오류가 발생했습니다."),
    BIND_ERROR(HttpStatus.BAD_REQUEST, "요청 파라미터 바인딩에 실패했습니다."),
    ARGUMENT_TYPE_MISMATCH(HttpStatus.BAD_REQUEST, "요청 파라미터 타입이 일치하지 않습니다."),

    // 회원
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "사용자를 찾을 수 없습니다."),
    INVALID_USER(HttpStatus.FORBIDDEN, "접근 권한이 없습니다."),

    // 인증
    INVALID_TOKEN(HttpStatus.UNAUTHORIZED, "유효하지 않은 토큰입니다."),
    AUTH_NOT_FOUND(HttpStatus.UNAUTHORIZED, "인증 정보가 없습니다."),
    EXPIRED_TOKEN(HttpStatus.UNAUTHORIZED, "토큰이 만료되었습니다."),

    // 프로젝트
    PROJECT_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 프로젝트를 찾을 수 없습니다."),
    PROJECT_MISMATCH(HttpStatus.FORBIDDEN, "현재 유저와 프로젝트 입력자가 일치하지 않습니다."),
    PROJECT_DATE_NOT_VALID(HttpStatus.BAD_REQUEST, "프로젝트 시작 날짜는 종료 날짜보다 먼저여야 합니다."),

    // 경험
    EXPERIENCE_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 경험을 찾을 수 없습니다."),
    EXPERIENCE_MISMATCH(HttpStatus.FORBIDDEN, "현재 유저와 경험 입력자가 일치하지 않습니다."),
    EXPERIENCE_DATE_NOT_VALID(HttpStatus.BAD_REQUEST, "경험 시작 날짜는 종료 날짜보다 먼저여야 합니다."),

    // 직무 분석
    JOB_ROLE_ANALYSIS_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 직무 분석 레포트를 찾을 수 없습니다."),
    JOB_ROLE_ANALYSIS_BOOKMARK_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 즐겨찾기 항목을 찾을 수 없습니다."),

    // 기업 분석
    COMPANY_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 기업 정보를  찾을 수 없습니다."),
    COMPANY_ANALYSIS_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 기업 분석 레포트를  찾을 수 없습니다."),
    COMPANY_ANALYSIS_ALREADY_BOOKMARK(HttpStatus.NOT_FOUND, "이미 즐겨찾기에 추가된 항목입니다."),
    COMPANY_ANALYSIS_BOOKMARK_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 즐겨찾기 항목을 찾을 수 없습니다."),
    COMPANY_ANALYSIS_REQUEST_LIMIT_EXCEEDED(HttpStatus.NO_CONTENT, "일일 기업 분석 요청 횟수를 초과하였습니다."),
    FASTAPI_RESPONSE_NULL(HttpStatus.BAD_REQUEST, "fast API 반환값이 없습니다"),


    // 자기소개서 관련
    JOB_ROLE_SNAPSHOT_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 직무 분석 스냅샷을 찾을 수 없습니다."),
    COVER_LETTER_CONTENT_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 자기소개서 본문을 찾을 수 없습니다."),
    COVER_LETTER_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 자기소개서를 찾을 수 없습니다."),
    COVER_LETTER_MISMATCH(HttpStatus.FORBIDDEN, "현재 유저와 자기소개서 작성자가 일치하지 않습니다."),
    COVER_LETTER_CONTENT_ALREADY_START(HttpStatus.BAD_REQUEST, "이미 작성 중이거나 작성 완료된 자기소개서 문항입니다."),

    // 일정
    SCHEDULE_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 일정을 찾을 수 없습니다."),
    COVER_LETTER_ALREADY_IN_USE(HttpStatus.BAD_REQUEST, "이미 해당 자기소개서로 등록된 일정이 있습니다."),
    SCHEDULE_STATUS_NOT_FOUND(HttpStatus.NOT_FOUND, "해당 상태를 찾을 수 없습니다.")


    /**
     Response의 에러 코드에 맞춰 HttpStatus를 설정해주시기 바랍니다.

     // fail
     BAD_REQUEST(400)
     UNAUTHORIZED(401)
     FORBIDDEN(403)
     NOT_FOUND(404)
     METHOD_NOT_ALLOWED(405)
     INTERNAL_SERVER_ERROR(500)

     **/

    ;

    private final HttpStatus httpStatus;
    private final String message;
}
