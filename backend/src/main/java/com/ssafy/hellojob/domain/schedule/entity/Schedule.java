package com.ssafy.hellojob.domain.schedule.entity;

import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.sql.Date;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "schedule")
public class Schedule {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "schedule_id", nullable = false)
    private Integer scheduleId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "schedule_status_id", nullable = false)
    private ScheduleStatus scheduleStatus;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "cover_letter_id", nullable = true)
    private CoverLetter coverLetter;

    @Column(name="schedule_start_date")
    private Date scheduleStartDate;

    @Column(name="schedule_end_date")
    private Date scheduleEndDate;

    @Column(name = "schedule_title", nullable = false, length = 100)
    private String scheduleTitle;

    @Column(name = "schedule_memo", length = 500)
    private String scheduleMemo;

    @Builder
    public Schedule(User user, ScheduleStatus scheduleStatus, CoverLetter coverLetter, Date scheduleStartDate, Date scheduleEndDate, String scheduleTitle, String scheduleMemo){
        this.user = user;
        this.scheduleStatus = scheduleStatus;
        this.coverLetter = coverLetter;
        this.scheduleStartDate=scheduleStartDate;
        this.scheduleEndDate = scheduleEndDate;
        this.scheduleTitle = scheduleTitle;
        this.scheduleMemo = scheduleMemo;

    }

    public void setScheduleStatus(ScheduleStatus scheduleStatus){
        this.scheduleStatus = scheduleStatus;
    }

    public void setScheduleCoverLetter(CoverLetter coverLetter){
        this.coverLetter = coverLetter;
    }

    public void setScheduleStartDate(Date scheduleStartDate){
        this.scheduleStartDate = scheduleStartDate;
    }

    public void setScheduleEndDate(Date scheduleEndDate){
        this.scheduleEndDate = scheduleEndDate;
    }

    public void setScheduleTitle(String scheduleTitle){
        this.scheduleTitle = scheduleTitle;
    }

    public void setScheduleMemo(String scheduleMemo){
        this.scheduleMemo = scheduleMemo;
    }

}
