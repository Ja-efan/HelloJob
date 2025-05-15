package com.ssafy.hellojob.domain.coverletter.service;

import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.repository.CompanyAnalysisRepository;
import com.ssafy.hellojob.domain.coverletter.dto.request.CoverLetterRequestDto;
import com.ssafy.hellojob.domain.coverletter.dto.response.*;
import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ContentQuestionStatusDto;
import com.ssafy.hellojob.domain.coverletter.dto.response.CoverLetterStatusesDto;
import com.ssafy.hellojob.domain.coverlettercontent.service.CoverLetterContentService;
import com.ssafy.hellojob.domain.jobrolesnapshot.entity.JobRoleSnapshot;
import com.ssafy.hellojob.domain.coverletter.repository.CoverLetterRepository;
import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysis;
import com.ssafy.hellojob.domain.jobroleanalysis.repository.JobRoleAnalysisRepository;
import com.ssafy.hellojob.domain.jobrolesnapshot.service.JobRoleSnapshotService;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@Transactional
@RequiredArgsConstructor
public class CoverLetterService {

    private final CoverLetterRepository coverLetterRepository;
    private final CompanyAnalysisRepository companyAnalysisRepository;
    private final JobRoleAnalysisRepository jobRoleAnalysisRepository;
    private final JobRoleSnapshotService jobRoleSnapshotService;
    private final CoverLetterContentService coverLetterContentService;

    public CoverLetterCreateResponseDto createCoverLetter(User user, CoverLetterRequestDto requestDto) {
        CompanyAnalysis companyAnalysis = companyAnalysisRepository.findById(requestDto.getCompanyAnalysisId())
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_ANALYSIS_NOT_FOUND));

        String companyName = companyAnalysis.getCompany().getCompanyName();

        JobRoleSnapshot jobRoleSnapshot;

        if (requestDto.getJobRoleAnalysisId() == null) {
            jobRoleSnapshot = null;
        } else {
            JobRoleAnalysis jobRoleAnalysis = jobRoleAnalysisRepository.findById(requestDto.getJobRoleAnalysisId())
                    .orElseThrow(() -> new BaseException(ErrorCode.JOB_ROLE_ANALYSIS_NOT_FOUND));

            jobRoleSnapshot = jobRoleSnapshotService.copyJobRoleAnalysis(companyName, jobRoleAnalysis);
        }

        CoverLetter newCoverLetter = CoverLetter.builder()
                .coverLetterTitle(requestDto.getCoverLetterTitle())
                .user(user)
                .companyAnalysis(companyAnalysis)
                .jobRoleSnapshot(jobRoleSnapshot)
                .build();

        coverLetterRepository.save(newCoverLetter);

        Integer newCoverLetterId = newCoverLetter.getCoverLetterId();

        Integer firstContentId = coverLetterContentService.createContents(user, newCoverLetter, requestDto.getContents());

        return CoverLetterCreateResponseDto.builder()
                .coverLetterId(newCoverLetterId)
                .firstContentId(firstContentId)
                .build();
    }

    // 자기소개서 전체 문항 상태 조회
    public CoverLetterStatusesDto getCoverLetterStatuses(User user, Integer coverLetterId) {

        CoverLetter coverLetter = coverLetterRepository.findById(coverLetterId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

        if (!user.getUserId().equals(coverLetter.getUser().getUserId())){
            throw new BaseException(ErrorCode.COVER_LETTER_MISMATCH);
        }

        List<ContentQuestionStatusDto> contentQuestionStatuses = coverLetterContentService.getCoverLetterContentQuestionStatues(coverLetterId);
        int totalContentQuestionCount = contentQuestionStatuses.size();

        CoverLetterStatusesDto coverLetterStatusesDto = CoverLetterStatusesDto.builder()
                .coverLetterId(coverLetter.getCoverLetterId())
                .totalContentQuestionCount(totalContentQuestionCount)
                .contentQuestionStatuses(contentQuestionStatuses)
                .updatedAt(coverLetter.getUpdatedAt())
                .build();

        return coverLetterStatusesDto;
    }

    // 자기소개서 요약 조회
    public CoverLetterSummaryDto getCoverLetterSummary(User user, Integer coverLetterId) {
        CoverLetter coverLetter = coverLetterRepository.findById(coverLetterId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

        if (!user.getUserId().equals(coverLetter.getUser().getUserId())){
            throw new BaseException(ErrorCode.COVER_LETTER_MISMATCH);
        }

        List<Integer> contentIds = coverLetterContentService.getContentIdsByCoverLetterId(coverLetterId);

        return CoverLetterSummaryDto.builder()
                .coverLetterTitle(coverLetter.getCoverLetterTitle())
                .contentIds(contentIds)
                .companyAnalysisId(coverLetter.getCompanyAnalysis().getCompanyAnalysisId().intValue())
                .jobRoleSnapshotId(coverLetter.getJobRoleSnapshot().getJobRoleSnapshotId())
                .build();
    }

    public Map<String, String> saveAll(User user, Integer coverLetterId) {
        CoverLetter coverLetter = coverLetterRepository.findById(coverLetterId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

        if (!user.getUserId().equals(coverLetter.getUser().getUserId())){
            throw new BaseException(ErrorCode.COVER_LETTER_MISMATCH);
        }

        coverLetterContentService.saveAllContents(coverLetter);
        coverLetter.updateFinish(true);

        return Map.of("message", "자기소개서가 전체 저장되었습니다.");
    }

    // 자기소개서 삭제: 관련 엔터티 cascade로 전부 삭제
    public Map<String, String> deleteCoverLetter(User user, Integer coverLetterId) {
        CoverLetter coverLetter = coverLetterRepository.findById(coverLetterId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

        if (!coverLetter.getUser().getUserId().equals(user.getUserId())) {
            throw new BaseException(ErrorCode.COVER_LETTER_MISMATCH);
        }

        coverLetterRepository.delete(coverLetter);
        return Map.of("message", "자기소개서가 삭제되었습니다.");
    }

    // 마이페이지 자기소개서 목록 조회
    public Page<MyPageCoverLetterDto> getCoverLettersForMaPage(Integer userId, Pageable pageable) {
        Page<MyPageCoverLetterDto> list = coverLetterRepository.getCoverLettersByUser(userId, pageable);
        return list;
    }
}
