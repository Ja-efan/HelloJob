package com.ssafy.hellojob.domain.schedule.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.DartAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.NewsAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.repository.CompanyAnalysisRepository;
import com.ssafy.hellojob.domain.companyanalysis.repository.DartAnalysisRepository;
import com.ssafy.hellojob.domain.companyanalysis.repository.NewsAnalysisRepository;
import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.coverletter.repository.CoverLetterRepository;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.coverlettercontent.repository.CoverLetterContentRepository;
import com.ssafy.hellojob.domain.jobrolesnapshot.entity.JobRoleSnapshot;
import com.ssafy.hellojob.domain.jobrolesnapshot.repository.JobRoleSnapshotRepository;
import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleAddRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleUpdateScheduleCoverLetterRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleUpdateScheduleStatusRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.response.*;
import com.ssafy.hellojob.domain.schedule.entity.Schedule;
import com.ssafy.hellojob.domain.schedule.entity.ScheduleStatus;
import com.ssafy.hellojob.domain.schedule.repository.ScheduleRepository;
import com.ssafy.hellojob.domain.schedule.repository.ScheduleStatusRepository;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class ScheduleService {

    private final UserRepository userRepository;
    private final ScheduleRepository scheduleRepository;
    private final CoverLetterRepository coverLetterRepository;
    private final CoverLetterContentRepository coverLetterContentRepository;
    private final ScheduleStatusRepository scheduleStatusRepository;
    private final CompanyAnalysisRepository companyAnalysisRepository;
    private final DartAnalysisRepository dartAnalysisRepository;
    private final NewsAnalysisRepository newsAnalysisRepository;
    private final JobRoleSnapshotRepository jobRoleSnapshotRepository;


    // 일정 추가
    public ScheduleIdResponseDto addSchedule(ScheduleAddRequestDto requestDto, Integer userId){

        // 유저 정보 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        ScheduleStatus scheduleStatus = scheduleStatusRepository.findByScheduleStatusName(requestDto.getScheduleStatusName())
                .orElseThrow(()-> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));


        CoverLetter coverLetter = null;
        if(requestDto.getCoverLetterId() != null){
            coverLetter = coverLetterRepository.getReferenceById(requestDto.getCoverLetterId());
        }

        Schedule newSchedule = Schedule.builder()
                .user(user)
                .scheduleStatus(scheduleStatus)
                .coverLetter(coverLetter)
                .scheduleStartDate(requestDto.getScheduleStartDate())
                .scheduleEndDate(requestDto.getScheduleEndDate())
                .scheduleTitle(requestDto.getScheduleTitle())
                .scheduleMemo(requestDto.getScheduleMemo())
                .build();

        scheduleRepository.save(newSchedule);

        return new ScheduleIdResponseDto().builder()
                .scheduleId(newSchedule.getScheduleId())
                .build();

    }

    // 일정 삭제
    public void deleteSchedule(Integer scheduleId, Integer userId){

        // 유저 정보 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 스케줄 정보 조회
        Schedule schedule = scheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));

        // 작성자와 userId가 같을 때만 삭제
        if(userId == schedule.getUser().getUserId()){
            scheduleRepository.delete(schedule);
        } else {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

    }

    // 일정 상태 수정
    public ScheduleIdResponseDto updateScheduleStatus(ScheduleUpdateScheduleStatusRequestDto requestDto, Integer scheduleId, Integer userId){
        // 유저 정보 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 스케줄 정보 조회
        Schedule schedule = scheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));

        // 권한 검증 (스케줄 소유자와 일치하는지 확인)
        if (!schedule.getUser().getUserId().equals(userId)) {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        // 새로운 상태 조회
        ScheduleStatus newStatus = scheduleStatusRepository.findByScheduleStatusName(requestDto.getScheduleStatusName())
                .orElseThrow(()-> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));
        if (newStatus == null) {
            throw new BaseException(ErrorCode.SCHEDULE_STATUS_NOT_FOUND);
        }

        // 상태 변경
        schedule.setScheduleStatus(newStatus);
        scheduleRepository.save(schedule);

        // 저장
        return new ScheduleIdResponseDto(schedule.getScheduleId());

    }

    // 일정 자기소개서 수정
    public ScheduleIdResponseDto updateScheduleCoverLetter(ScheduleUpdateScheduleCoverLetterRequestDto requestDto, Integer scheduleId, Integer userId){
        // 유저 정보 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 스케줄 정보 조회
        Schedule schedule = scheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));

        // 권한 검증 (스케줄 소유자와 일치하는지 확인)
        if (!schedule.getUser().getUserId().equals(userId)) {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        // 자기소개서 조회
        CoverLetter coverLetter = coverLetterRepository.getReferenceById(requestDto.getCoverLetterId());
        if(coverLetter == null){
            throw new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND);
        }

        // 상태 변경
        schedule.setScheduleCoverLetter(coverLetter);
        scheduleRepository.save(schedule);

        // 저장
        return new ScheduleIdResponseDto(schedule.getScheduleId());

    }

    // 일정 전체 수정
    @Transactional
    public ScheduleIdResponseDto updateSchedule(ScheduleAddRequestDto requestDto, Integer scheduleId, Integer userId) {
        // 유저 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 기존 스케줄 조회
        Schedule schedule = scheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));

        if (!schedule.getUser().getUserId().equals(userId)) {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        // 상태 값 수정
        ScheduleStatus status = scheduleStatusRepository.findByScheduleStatusName(requestDto.getScheduleStatusName())
                .orElseThrow(()-> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));
        if (status == null) {
            throw new BaseException(ErrorCode.SCHEDULE_STATUS_NOT_FOUND);
        }
        schedule.setScheduleStatus(status);

        // CoverLetter는 요청에 값이 있을 경우에만 수정
        if (requestDto.getCoverLetterId() != null) {
            CoverLetter coverLetter = coverLetterRepository.findById(requestDto.getCoverLetterId())
                    .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

            // 동일한 커버레터를 다시 설정하는 경우에는 생략
            if (schedule.getCoverLetter() == null || !schedule.getCoverLetter().getCoverLetterId().equals(coverLetter.getCoverLetterId())) {
                // 커버레터가 이미 다른 스케줄에 연결되어 있는지 확인
                Optional<Schedule> otherSchedule = scheduleRepository.findByCoverLetter(coverLetter);
                if (otherSchedule.isPresent() && !otherSchedule.get().getScheduleId().equals(scheduleId)) {
                    throw new BaseException(ErrorCode.COVER_LETTER_ALREADY_IN_USE); // 커스텀 에러 코드 정의 필요
                }
                schedule.setScheduleCoverLetter(coverLetter);
            }
        }

        schedule.setScheduleStartDate(requestDto.getScheduleStartDate());
        schedule.setScheduleEndDate(requestDto.getScheduleEndDate());
        schedule.setScheduleTitle(requestDto.getScheduleTitle());
        schedule.setScheduleMemo(requestDto.getScheduleMemo());

        return new ScheduleIdResponseDto(schedule.getScheduleId());
    }

    // 일정 전체 조회
    public List<ScheduleListResponseDto> allSchedule(Integer userId){

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        List<Schedule> schedules = scheduleRepository.findByUser(user);

        List<ScheduleListResponseDto> responseDto = new ArrayList<>();

        for(Schedule schedule: schedules){
            responseDto.add(ScheduleListResponseDto.builder()
                            .scheduleId(schedule.getScheduleId())
                            .scheduleStartDate(schedule.getScheduleStartDate())
                            .scheduleEndDate(schedule.getScheduleEndDate())
                            .scheduleTitle(schedule.getScheduleTitle())
                            .scheduleStatusName(schedule.getScheduleStatus().getScheduleStatusName())
                            .scheduleStatusStep(schedule.getScheduleStatus().getScheduleStatusStep().name())
                    .build());
        }

        return responseDto;
    }

    // 일정 상세 조회
    public ScheduleDetailResponseDto detailSchedule(Integer scheduleId, Integer userId) {

        userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        Schedule schedule = scheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new BaseException(ErrorCode.SCHEDULE_NOT_FOUND));

        ScheduleDetailCoverLetter scheduleDetailCoverLetter = null;
        ScheduleDetailCompanyAnalysis scheduleDetailCompanyAnalysis = null;
        ScheduleDetailJobRoleSnapshot scheduleDetailJobRoleSnapshot = null;

        if (!schedule.getUser().getUserId().equals(userId)) {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        if (schedule.getCoverLetter() != null) {
            CoverLetter coverLetter = coverLetterRepository.getReferenceById(schedule.getCoverLetter().getCoverLetterId());

            // 자기소개서 내용 구성
            List<CoverLetterContent> coverLetterContents = coverLetterContentRepository.findByCoverLetter(coverLetter);
            List<ScheduleCoverLetterContent> scheduleCoverLetterContents = coverLetterContents.stream()
                    .map(content -> ScheduleCoverLetterContent.builder()
                            .contentId(content.getContentId())
                            .contentNumber(content.getContentNumber())
                            .contentQuestion(content.getContentQuestion())
                            .contentLength(content.getContentLength())
                            .contentDetail(content.getContentDetail())
                            .build())
                    .toList();

            scheduleDetailCoverLetter = ScheduleDetailCoverLetter.builder()
                    .coverLetterTitle(coverLetter.getCoverLetterTitle())
                    .coverLetterId(coverLetter.getCoverLetterId())
                    .finish(coverLetter.isFinish())
                    .scheduleCoverLetterContents(scheduleCoverLetterContents)
                    .build();

            // 기업 분석
            CompanyAnalysis companyAnalysis = companyAnalysisRepository.getReferenceById(coverLetter.getCompanyAnalysis().getCompanyAnalysisId());
            DartAnalysis dartAnalysis = dartAnalysisRepository.getReferenceById(companyAnalysis.getDartAnalysis().getDartAnalysisId());
            NewsAnalysis newsAnalysis = newsAnalysisRepository.getReferenceById(companyAnalysis.getNewsAnalysis().getNewsAnalysisId());

            List<String> dartCategory = new ArrayList<>();
            if (dartAnalysis.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
            if (dartAnalysis.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
            if (dartAnalysis.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");

            List<String> newsUrls = new ArrayList<>();
            if (newsAnalysis.getNewsAnalysisUrl() != null && !newsAnalysis.getNewsAnalysisUrl().isBlank()) {
                try {
                    ObjectMapper objectMapper = new ObjectMapper();
                    newsUrls = objectMapper.readValue(newsAnalysis.getNewsAnalysisUrl(), new TypeReference<List<String>>() {});
                } catch (Exception e) {
                    throw new RuntimeException("뉴스 URL 파싱 실패", e);
                }
            }

            scheduleDetailCompanyAnalysis = ScheduleDetailCompanyAnalysis.builder()
                    .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                    .companyName(companyAnalysis.getCompany().getCompanyName())
                    .newsAnalysisData(newsAnalysis.getNewsAnalysisData())
                    .newsAnalysisUrl(newsUrls)
                    .dartBrand(dartAnalysis.getDartBrand())
                    .dartVision(dartAnalysis.getDartVision())
                    .dartCompanyAnalysis(dartAnalysis.getDartCompanyAnalysis())
                    .dartFinancialSummery(dartAnalysis.getDartFinancialSummary())
                    .dartCategory(dartCategory)
                    .build();

            // 직무 스냅샷
            if (coverLetter.getJobRoleSnapshot() != null) {
                JobRoleSnapshot jobRoleSnapshot = jobRoleSnapshotRepository.getReferenceById(
                        coverLetter.getJobRoleSnapshot().getJobRoleSnapshotId());

                scheduleDetailJobRoleSnapshot = ScheduleDetailJobRoleSnapshot.builder()
                        .jobRoleSnapshotId(jobRoleSnapshot.getJobRoleSnapshotId())
                        .jobRoleAnalysisId(jobRoleSnapshot.getJobRoleAnalysisId())
                        .jobRoleSnapshotCategory(jobRoleSnapshot.getJobRoleSnapshotCategory())
                        .jobRoleSnapshotName(jobRoleSnapshot.getJobRoleSnapshotName())
                        .jobRoleSnapshotTitle(jobRoleSnapshot.getJobRoleSnapshotTitle())
                        .jobRoleSnapshotSkills(jobRoleSnapshot.getJobRoleSnapshotSkills())
                        .jobRoleSnapshotEtc(jobRoleSnapshot.getJobRoleSnapshotEtc())
                        .jobRoleSnapshotWork(jobRoleSnapshot.getJobRoleSnapshotWork())
                        .jobRoleSnapshotRequirements(jobRoleSnapshot.getJobRoleSnapshotRequirements())
                        .jobRoleSnapshotPreferences(jobRoleSnapshot.getJobRoleSnapshotPreferences())
                        .build();
            }
        }

        return ScheduleDetailResponseDto.builder()
                .scheduleId(scheduleId)
                .scheduleStartDate(schedule.getScheduleStartDate())
                .scheduleEndDate(schedule.getScheduleEndDate())
                .scheduleTitle(schedule.getScheduleTitle())
                .scheduleMemo(schedule.getScheduleMemo())
                .scheduleStatusName(schedule.getScheduleStatus().getScheduleStatusName())
                .scheduleStatusStep(schedule.getScheduleStatus().getScheduleStatusStep().name())
                .scheduleDetailCoverLetter(scheduleDetailCoverLetter)
                .scheduleDetailCompanyAnalysis(scheduleDetailCompanyAnalysis)
                .scheduleDetailJobRoleSnapshot(scheduleDetailJobRoleSnapshot)
                .build();
    }


}
