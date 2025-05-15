package com.ssafy.hellojob.domain.companyanalysis.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.sql.Date;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "news_analysis")
public class NewsAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "news_analysis_id", nullable = false)
    private Integer newsAnalysisId;

    @Column(name = "news_analysis_data", nullable = false, columnDefinition = "TEXT")
    private String newsAnalysisData;

    @Column(name = "news_analysis_date")
    private Date newsAnalysisDate;

    @Column(name = "news_analysis_url", columnDefinition = "TEXT")
    private String newsAnalysisUrl;

    public static NewsAnalysis of(String summary, Date date, String urlJson) {
        NewsAnalysis news = new NewsAnalysis();
        news.newsAnalysisData = summary;
        news.newsAnalysisDate = date;
        news.newsAnalysisUrl = urlJson;
        return news;
    }

}
