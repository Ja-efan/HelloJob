package com.ssafy.hellojob.domain.companyanalysis.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.hellojob.domain.company.entity.Company;
import com.ssafy.hellojob.domain.company.repository.CompanyRepository;
import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisBookmarkSaveRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.response.*;
import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysisBookmark;
import com.ssafy.hellojob.domain.companyanalysis.entity.DartAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.NewsAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.repository.CompanyAnalysisBookmarkRepository;
import com.ssafy.hellojob.domain.companyanalysis.repository.CompanyAnalysisRepository;
import com.ssafy.hellojob.domain.companyanalysis.repository.DartAnalysisRepository;
import com.ssafy.hellojob.domain.companyanalysis.repository.NewsAnalysisRepository;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CompanyAnalysisService {

    private final CompanyRepository companyRepository;
    private final CompanyAnalysisRepository companyAnalysisRepository;
    private final CompanyAnalysisBookmarkRepository companyAnalysisBookmarkRepository;
    private final DartAnalysisRepository dartAnalysisRepository;
    private final NewsAnalysisRepository newsAnalysisRepository;
    private final UserRepository userRepository;

    // 기업 분석 저장
    @Transactional
    public CompanyAnalysisBookmarkSaveRequestDto createCompanyAnalysis(Integer userId,
                                                                       CompanyAnalysisRequestDto requestDto,
                                                                       CompanyAnalysisFastApiResponseDto responseDto) {
        // 유저 조회 및 토큰(기업 분석 가능 횟수) 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        if (user.getToken() <= 0) {
            throw new BaseException(ErrorCode.COMPANY_ANALYSIS_REQUEST_LIMIT_EXCEEDED);
        }

        // 토큰 감소
        user.decreaseToken();

        // 회사 조회
        Company company = companyRepository.findById(requestDto.getCompanyId())
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_NOT_FOUND));

        // DartAnalysis 저장
        DartAnalysis dart = DartAnalysis.of(
                responseDto.getCompany_brand(),
                responseDto.getCompany_analysis(),
                responseDto.getCompany_vision(),
                responseDto.getCompany_finance(),
                requestDto.isBasic(),
                requestDto.isPlus(),
                requestDto.isFinancial()
        );

        dartAnalysisRepository.save(dart);

        // NewsAnalysis 저장
        String jsonUrls;
        try {
            jsonUrls = new ObjectMapper().writeValueAsString(responseDto.getNews_urls());
        } catch (JsonProcessingException e) {
            throw new RuntimeException("뉴스 URL 직렬화 실패", e);
        }

        NewsAnalysis news = NewsAnalysis.of(
                responseDto.getNews_summary(),
                responseDto.getAnalysis_date(),
                jsonUrls
        );

        newsAnalysisRepository.save(news);

        // CompanyAnalysis 저장
        CompanyAnalysis companyAnalysis = CompanyAnalysis.of(user, company, dart, news, requestDto.isPublic());
        companyAnalysisRepository.save(companyAnalysis);

        // 기업 테이블 업데이트
        company.setUpdatedAt(LocalDateTime.now());

        return CompanyAnalysisBookmarkSaveRequestDto.builder()
                .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                .build();
    }


    // 기업 분석 목록 전체 조회
    public List<CompanyAnalysisListResponseDto> searchAllCompanyAnalysis(Integer userId) {
        List<CompanyAnalysis> analysisList = companyAnalysisRepository.findAll();

        List<CompanyAnalysisListResponseDto> result = analysisList.stream()
                .filter(CompanyAnalysis::isPublic) // 공개된 기업 분석만 조회
                .map(analysis -> {
                    DartAnalysis dart = analysis.getDartAnalysis();
                    List<String> dartCategory = new ArrayList<>();
                    if (dart != null) {
                        if (dart.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
                        if (dart.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
                        if (dart.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");
                    }

                    return CompanyAnalysisListResponseDto.builder()
                            .companyAnlaysisId(analysis.getCompanyAnalysisId())
                            .companyName(analysis.getCompany().getCompanyName())
                            .createdAt(analysis.getCreatedAt())
                            .companyViewCount(analysis.getCompanyAnalysisViewCount())
                            .companyLocation(analysis.getCompany().getCompanyLocation())
                            .companySize(analysis.getCompany().getCompanySize().name())
                            .companyIndustry(analysis.getCompany().getCompanyIndustry())
                            .companyAnalysisBookmarkCount(analysis.getCompanyAnalysisBookmarkCount())
                            .bookmark(companyAnalysisBookmarkRepository.existsByUser_UserIdAndCompanyAnalysis_CompanyAnalysisId(
                                    userId, analysis.getCompanyAnalysisId()))
                            .isPublic(analysis.isPublic())
                            .dartCategory(dartCategory)
                            .build();
                })
                .toList();

        return result;
    }


    // 기업 분석 상세 조회
    @Transactional
    public CompanyAnalysisDetailResponseDto detailCompanyAnalysis(Integer userId, Integer companyAnalysisId) {

        // 유저 조회
        userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 기업 분석 데이터 조회
        CompanyAnalysis companyAnalysis = companyAnalysisRepository.findById(companyAnalysisId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_ANALYSIS_NOT_FOUND));

        // 공개 여부 필터링
        if (!companyAnalysis.isPublic()) {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        // 조회수 증가
        companyAnalysis.setCompanyAnalysisViewCount(companyAnalysis.getCompanyAnalysisViewCount() + 1);
        companyAnalysisRepository.save(companyAnalysis);

        // 즐겨찾기 여부 필터링
        boolean isBookmarked = companyAnalysisBookmarkRepository.existsByUser_UserIdAndCompanyAnalysis_CompanyAnalysisId(userId, companyAnalysisId);

        // dart 분석 시 사용된 데이터 배열로 변환
        DartAnalysis dart = companyAnalysis.getDartAnalysis();
        List<String> dartCategory = new ArrayList<>();
        if (dart.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
        if (dart.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
        if (dart.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");

        // 해당 기업 분석에 활용된 뉴스 분석 정보 불러오기
        NewsAnalysis news = companyAnalysis.getNewsAnalysis();

        // 뉴스 크롤링 출처 기사 링크 배열로 변환
        List<String> newsUrls = new ArrayList<>();
        if (news.getNewsAnalysisUrl() != null && !news.getNewsAnalysisUrl().isBlank()) {
            try {
                ObjectMapper objectMapper = new ObjectMapper();
                newsUrls = objectMapper.readValue(news.getNewsAnalysisUrl(), new TypeReference<List<String>>() {});
            } catch (Exception e) {
                throw new RuntimeException("뉴스 URL 파싱 실패", e);
            }
        }

        return CompanyAnalysisDetailResponseDto.builder()
                .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                .companyName(companyAnalysis.getCompany().getCompanyName())
                .createdAt(companyAnalysis.getCreatedAt())
                .companyViewCount(companyAnalysis.getCompanyAnalysisViewCount())
                .companyLocation(companyAnalysis.getCompany().getCompanyLocation())
                .companySize(companyAnalysis.getCompany().getCompanySize().name())
                .companyIndustry(companyAnalysis.getCompany().getCompanyIndustry())
                .companyAnalysisBookmarkCount(companyAnalysis.getCompanyAnalysisBookmarkCount())
                .bookmark(isBookmarked)
                .isPublic(companyAnalysis.isPublic())
                .newsAnalysisData(news.getNewsAnalysisData())
                .newsAnalysisDate(news.getNewsAnalysisDate())
                .newsAnalysisUrl(newsUrls) 
                .dartBrand(dart.getDartBrand())
                .dartCompanyAnalysis(dart.getDartCompanyAnalysis())
                .dartVision(dart.getDartVision())
                .dartFinancialSummery(dart.getDartFinancialSummary())
                .dartCategory(dartCategory)
                .build();
    }

    // 기업 분석 검색
    public List<CompanyAnalysisListResponseDto> searchByCompanyIdCompanyAnalysis(Integer companyId, Integer userId) {

        // 유저, 회사 존재 여부 확인
        userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));
        companyRepository.findById(companyId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_NOT_FOUND));

        // 해당 기업의 기업 분석 전체 조회
        List<CompanyAnalysis> analysisList = companyAnalysisRepository.findAllByCompany_CompanyId(companyId);

        // 공개된 분석만 필터링하여 DTO 매핑
        return analysisList.stream()
                .filter(CompanyAnalysis::isPublic)
                .map(analysis -> {
                    DartAnalysis dart = analysis.getDartAnalysis();
                    List<String> dartCategory = new ArrayList<>();
                    if (dart != null) {
                        if (dart.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
                        if (dart.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
                        if (dart.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");
                    }

                    return CompanyAnalysisListResponseDto.builder()
                            .companyAnlaysisId(analysis.getCompanyAnalysisId())
                            .companyName(analysis.getCompany().getCompanyName())
                            .createdAt(analysis.getCreatedAt())
                            .companyViewCount(analysis.getCompanyAnalysisViewCount())
                            .companyLocation(analysis.getCompany().getCompanyLocation())
                            .companySize(analysis.getCompany().getCompanySize().name())
                            .companyIndustry(analysis.getCompany().getCompanyIndustry())
                            .companyAnalysisBookmarkCount(analysis.getCompanyAnalysisBookmarkCount())
                            .bookmark(companyAnalysisBookmarkRepository
                                    .existsByUser_UserIdAndCompanyAnalysis_CompanyAnalysisId(userId, analysis.getCompanyAnalysisId()))
                            .isPublic(analysis.isPublic())
                            .dartCategory(dartCategory)
                            .build();
                })
                .toList();
    }


    // 기업 분석 북마크 추가
    @Transactional
    public CompanyAnalysisBookmarkSaveResponseDto addCompanyAnalysisBookmark(Integer userId, CompanyAnalysisBookmarkSaveRequestDto requestDto) {

        // 기업 분석 데이터 조회
        CompanyAnalysis companyAnalysis = companyAnalysisRepository.findById(requestDto.getCompanyAnalysisId())
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_ANALYSIS_NOT_FOUND));

        // 유저 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 이미 북마크 되어 있는지 여부 확인
        boolean alreadyBookmarked = companyAnalysisBookmarkRepository.existsByUser_UserIdAndCompanyAnalysis_CompanyAnalysisId(userId, companyAnalysis.getCompanyAnalysisId());

        // 이미 되어 있으면 기존에 저장되어 있던 정보 반환(에러 처리x)
        if (alreadyBookmarked) {
            CompanyAnalysisBookmark existingBookmark = companyAnalysisBookmarkRepository.findByUserAndCompanyAnalysis(user, companyAnalysis)
                    .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_ANALYSIS_ALREADY_BOOKMARK));
            return CompanyAnalysisBookmarkSaveResponseDto.builder()
                    .companyAnalysisBookmarkId(existingBookmark.getCompanyAnalysisBookmarkId())
                    .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                    .build();
        }

        // 새 북마크 저장 객체 생성
        CompanyAnalysisBookmark newBookmark = CompanyAnalysisBookmark.builder()
                .user(user)
                .companyAnalysis(companyAnalysis)
                .build();

        // 북마크 정보 저장
        companyAnalysisBookmarkRepository.save(newBookmark);

        // 기업 분석 테이블 북마크 수 +1
        companyAnalysis.setCompanyAnalysisBookmarkCount(companyAnalysis.getCompanyAnalysisBookmarkCount() + 1);
        companyAnalysisRepository.save(companyAnalysis);

        return CompanyAnalysisBookmarkSaveResponseDto.builder()
                .companyAnalysisBookmarkId(newBookmark.getCompanyAnalysisBookmarkId())
                .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                .build();
    }

    // 기업 분석 북마크 해제
    @Transactional
    public void deleteCompanyAnalysisBookmark(Integer companyAnalysisBookmarkId, Integer userId){

        // 유저 조회
        userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 북마크 정보 조회
        CompanyAnalysisBookmark bookmark = companyAnalysisBookmarkRepository.findById(companyAnalysisBookmarkId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_ANALYSIS_BOOKMARK_NOT_FOUND));

        // 기업 분석 데이터 조회
        CompanyAnalysis companyAnalysis = bookmark.getCompanyAnalysis();

        // 북마크 정보의 userId와 요청한 userId가 같을 때만 요청 처리
        if(userId.equals(bookmark.getUser().getUserId())){
            companyAnalysisBookmarkRepository.delete(bookmark);
        } else {
            throw new BaseException(ErrorCode.INVALID_USER);
        }

        // 기업 분석 데이터 북마크 수 -1
        companyAnalysis.setCompanyAnalysisBookmarkCount(companyAnalysis.getCompanyAnalysisBookmarkCount() - 1);
        companyAnalysisRepository.save(companyAnalysis);
    }

    // 기업 분석 북마크 목록 조회(기업 상관 없이 전부)
    public List<CompanyAnalysisBookmarkListResponseDto> searchCompanyAnalysisBookmarkList(Integer userId) {
        
        // 유저 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));

        // 유저가 북마크한 모든 북마크 리스트 조회
        List<CompanyAnalysisBookmark> bookmarkList = companyAnalysisBookmarkRepository.findAllByUser(user);

        // 기업 분석 데이터 저장할 객체 배열 생성
        List<CompanyAnalysisBookmarkListResponseDto> result = new ArrayList<>();

        // 불러온 북마크 정보 기반으로 기업 분석 데이터 불러와서 result에 저장
        for (CompanyAnalysisBookmark bookmark : bookmarkList) {
            CompanyAnalysis companyAnalysis = bookmark.getCompanyAnalysis();

            // 공개 여부 처리(비공개일경우 pass)
            if (!companyAnalysis.isPublic()) {
                continue;
            }

            DartAnalysis dart = companyAnalysis.getDartAnalysis();
            List<String> dartCategory = new ArrayList<>();
            if (dart != null) {
                if (dart.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
                if (dart.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
                if (dart.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");
            }

            result.add(CompanyAnalysisBookmarkListResponseDto.builder()
                    .companyAnalysisBookmarkId(bookmark.getCompanyAnalysisBookmarkId())
                    .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                    .companyName(companyAnalysis.getCompany().getCompanyName())
                    .createdAt(companyAnalysis.getCreatedAt())
                    .companyViewCount(companyAnalysis.getCompanyAnalysisViewCount())
                    .companyLocation(companyAnalysis.getCompany().getCompanyLocation())
                    .companySize(companyAnalysis.getCompany().getCompanySize().name())
                    .companyIndustry(companyAnalysis.getCompany().getCompanyIndustry())
                    .companyAnalysisBookmarkCount(companyAnalysis.getCompanyAnalysisBookmarkCount())
                    .bookmark(true)
                    .isPublic(companyAnalysis.isPublic())
                    .dartCategory(dartCategory)
                    .build());
        }

        return result;
    }


    // 기업 분석 북마크 목록 조회(기업별)
    public List<CompanyAnalysisBookmarkListResponseDto> searchCompanyAnalysisBookmarkListWithCompanyId(Integer userId, Integer companyId) {
        
        // 유저, 회사 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(ErrorCode.USER_NOT_FOUND));
        companyRepository.findById(companyId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_NOT_FOUND));

        // 유저 + 기업 정보 기반 북마크 리스트 조회
        List<CompanyAnalysisBookmark> bookmarkList = companyAnalysisBookmarkRepository.findAllByUserAndCompanyAnalysis_Company_CompanyId(user, companyId);

        // 결과 반환할 객체 배열 생성
        List<CompanyAnalysisBookmarkListResponseDto> result = new ArrayList<>();

        // 불러온 북마크 정보 기반으로 기업 분석 데이터 불러와서 result에 저장
        for (CompanyAnalysisBookmark bookmark : bookmarkList) {
            CompanyAnalysis companyAnalysis = bookmark.getCompanyAnalysis();

            // 공개 여부 처리(비공개 시 pass)
            if (!companyAnalysis.isPublic()) {
                continue;
            }

            DartAnalysis dart = companyAnalysis.getDartAnalysis();
            List<String> dartCategory = new ArrayList<>();
            if (dart != null) {
                if (dart.isDartCompanyAnalysisBasic()) dartCategory.add("사업보고서 기본");
                if (dart.isDartCompanyAnalysisPlus()) dartCategory.add("사업보고서 상세");
                if (dart.isDartCompanyAnalysisFinancialData()) dartCategory.add("재무 정보");
            }

            result.add(CompanyAnalysisBookmarkListResponseDto.builder()
                    .companyAnalysisBookmarkId(bookmark.getCompanyAnalysisBookmarkId())
                    .companyAnalysisId(companyAnalysis.getCompanyAnalysisId())
                    .companyName(companyAnalysis.getCompany().getCompanyName())
                    .createdAt(companyAnalysis.getCreatedAt())
                    .companyViewCount(companyAnalysis.getCompanyAnalysisViewCount())
                    .companyLocation(companyAnalysis.getCompany().getCompanyLocation())
                    .companySize(companyAnalysis.getCompany().getCompanySize().name())
                    .companyIndustry(companyAnalysis.getCompany().getCompanyIndustry())
                    .companyAnalysisBookmarkCount(companyAnalysis.getCompanyAnalysisBookmarkCount())
                    .bookmark(true)
                    .isPublic(companyAnalysis.isPublic())
                    .dartCategory(dartCategory)
                    .build());
        }

        return result;
    }


}
