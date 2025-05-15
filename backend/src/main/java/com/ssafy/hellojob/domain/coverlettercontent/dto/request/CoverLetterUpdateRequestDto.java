package com.ssafy.hellojob.domain.coverlettercontent.dto.request;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class CoverLetterUpdateRequestDto {
    private String contentDetail;
    private CoverLetterContentStatus contentStatus;
}
