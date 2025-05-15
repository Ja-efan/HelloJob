package com.ssafy.hellojob.domain.user.service;

import com.ssafy.hellojob.domain.user.dto.response.ChangeNicknameResponseDto;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserReadService userReadService;
    private final UserRepository userRepository;


    public ChangeNicknameResponseDto changeNickname(String newNickname, Integer userId) {
        User user = userReadService.findUserByIdOrElseThrow(userId);
        user.changeNickname(newNickname);
        userRepository.save(user);
        return ChangeNicknameResponseDto.builder()
                .nickname(newNickname)
                .message("닉네임 변경이 완료되었습니다.")
                .build();
    }

}
