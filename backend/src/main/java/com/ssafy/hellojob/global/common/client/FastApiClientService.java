package com.ssafy.hellojob.global.common.client;

import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisFastApiRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.response.CompanyAnalysisFastApiResponseDto;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class FastApiClientService {

    private final WebClient fastApiWebClient;

    public CompanyAnalysisFastApiResponseDto sendJobAnalysisToFastApi(CompanyAnalysisFastApiRequestDto requestDto) {

        CompanyAnalysisFastApiResponseDto response = fastApiWebClient.post()
                .uri("/api/v1/ai/company-analysis")
                .bodyValue(requestDto)
                .retrieve()
                .bodyToMono(CompanyAnalysisFastApiResponseDto.class)
                .block();

        if (response == null) {
            throw new BaseException(ErrorCode.FASTAPI_RESPONSE_NULL);
        }

        return response;
    }

}
