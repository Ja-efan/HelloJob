package com.ssafy.hellojob.domain.company.entity;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum CompanySize {

    대기업("대기업"),
    중소기업("중소기업"),
    중견기업("중견기업"),
    정보없음("정보없음");

    private final String label;
}
