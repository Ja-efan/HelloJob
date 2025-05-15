package com.ssafy.hellojob.domain.jobroleanalysis.entity;

import com.fasterxml.jackson.annotation.JsonCreator;
import lombok.Getter;

@Getter
public enum JobRoleCategory {
    서버백엔드개발자,
    프론트엔드개발자,
    안드로이드개발자,
    iOS개발자,
    크로스플랫폼앱개발자,
    게임클라이언트개발자,
    게임서버개발자,
    DBA,
    빅데이터엔지니어,
    인공지능머신러닝,
    devops시스템엔지니어,
    정보보안침해대응,
    QA엔지니어,
    개발PM,
    HW펌웨어개발,
    SW솔루션,
    헬스테크,
    VRAR3D,
    블록체인,
    기술지원,
    기타;

    @JsonCreator
    public static JobRoleCategory from(String input) {
        if (input == null) {
            return null;
        }
        // 입력 문자열의 공백을 언더스코어로 변환해서 Enum 매칭
        String enumName = input.replace(" ", "");
        return JobRoleCategory.valueOf(enumName);
    }
}
