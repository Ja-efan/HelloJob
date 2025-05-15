package com.ssafy.hellojob.global.exception;


public final class ValidationMessage {

    private ValidationMessage() {}
    // Auth
    public static final String REFRESH_TOKEN_NOT_EMPTY = "refresh token은 필수값 입니다.";

    // User
    public static final String NICKNAME_NOT_EMPTY = "닉네임은 필수값 입니다.";
    public static final String ERROR_NICKNAME_LENGTH = "길이를 초과했습니다.";
    public static final String ERROR_NICKNAME_FORMAT = "이름은 숫자나 특수문자를 포함할 수 없습니다.";

    // Project
    public static final String PROJECT_NAME_NOT_EMPTY = "프로젝트명은 필수값 입니다.";
    public static final String PROJECT_INTRO_NOT_EMPTY = "프로젝트 개요는 필수값 입니다.";
    public static final String PROJECT_START_DATE_NOT_EMPTY = "프로젝트 시작일은 필수값 입니다.";
    public static final String PROJECT_END_DATE_NOT_EMPTY = "프로젝트 종료일은 필수값 입니다.";

    // Experience
    public static final String EXPERIENCE_NAME_NOT_EMPTY = "경험명은 필수값 입니다.";
    public static final String EXPERIENCE_DETAIL_NOT_EMPTY = "경험 상세 내역은 필수값 입니다.";
    public static final String EXPERIENCE_START_DATE_NOT_EMPTY = "경험 시작일은 필수값 입니다.";
    public static final String EXPERIENCE_END_DATE_NOT_EMPTY = "경험 종료일은 필수값 입니다.";

    // Job Role Analyis
    public static final String JOB_ROLE_ANALYSIS_COMPANY_ID_NOT_EMPTY = "기업 아이디는 필수값 입니다.";
    public static final String JOB_ROLE_ANALYSIS_JOB_ROLE_ANALYSIS_ID_NOT_EMPTY = "직무 분석 아이디는 필수값 입니다.";
    public static final String JOB_ROLE_ANALYSIS_JOB_ROLE_NAME_NOT_EMPTY = "직무명은 필수값 입니다.";
    public static final String JOB_ROLE_ANALYSIS_JOB_ROLE_CATEGORY_NOT_EMPTY = "직무 카테고리는 필수값 입니다.";

    // Company Analysis
    public static final String COMPANY_ANALYSIS_BOOKMARK_COMPANY_ANALYSIS_ID_NOT_EMPTY = "기업 분석 아이디는 필수값 입니다.";


    // Schedule
    public static final String SCHEDULE_TITLE_NOT_EMPTY = "스케쥴 제목은 필수값입니다.";
    public static final String SCHEDULE_STATUS_NOT_EMPTY = "스케줄 상태는 필수값입니다.";
    public static final String COVER_LETTER_ID_NOT_EMPTY = "자기소개서 아이디는 필수값입니다.";

}
