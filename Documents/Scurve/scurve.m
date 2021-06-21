function [t, y, dy, ddy, ta, tc] = fcn_s_curve(T0, T1, spd, acc, ts)
    dT = T1-T0;
    tc = [];
    
    if spd <= 0 || acc <= 0
       disp('error: negative speed, acc settings are not valid. Use positive values for all trajectories')
       return
    end
    
    if dT == 0
        disp('target-temp has to be different than the start temperature')
        return
    end
    
    if abs(dT) > spd^2/acc % constant speed trajectory
        
        % calculate phase times
        ta = spd/acc;
        tc = abs(dT)/spd - spd/acc;
        
        % calculate number of points per phase
        na = round(ta/ts);
        nc = round(tc/ts);
        n = 2*na+nc;
        
        % calculate rounded phase times
        tcr = nc*ts;
        tar = na*ts;
        tendr = n*ts;
        
        % adapt acc and speed at end of acc-phase to achieve dT exactly
        accr = abs(dT)/(tar*(tar+tcr));
        spdr = abs(dT)/(tar+tcr);
        
        % continuous acc, spd
        acc_cont = abs(dT)/(ta*(ta+tc));
        spd_cont = abs(dT)/(ta+tc);

        % create speed curve
        t = linspace(0,tendr,n);
        dy0 = linspace(0,spdr,na);
        dy1 = linspace(spdr,spdr,nc);
        dy2 = linspace(spdr,0,na);
        dy = [dy0,dy1,dy2]; 
    
    else % trajectory without constant speed
        
        % calculate phase time
        ta = sqrt(abs(dT)/acc);
        
        % calculate number of points per phase
        na = round(ta/ts);
        n = 2*na;
        
        % calculate rounded phase times
        tar = na*ts;  % time of acc after rounding
        tendr = 2*tar;
        
        % adapt acc and speed at end of acc-phase to achieve dT exactly
        accr = abs(dT)/tar^2;
        spdr = abs(dT)/tar; 
        
        % continuous acc, spd
        acc_cont = abs(dT)/ta^2;
        spd_cont = abs(dT)/ta; 
        
        % create speed curve
        t = linspace(0,tendr,n);
        dy0 = linspace(0,spdr,na);
        dy2 = linspace(spdr,0,na);
        dy = [dy0,dy2]; 
    end
    
    % change trajectory direction if start-temp is higher than target-temp
    dy = sign(dT)*dy;
    
    % check deviation from set values
    err_acc = abs( (acc_cont-accr)/acc_cont );
    err_spd = abs( (spd_cont-spdr)/spd_cont );
    fprintf('\n----------\nacc(max)=%.4f, acc(cont)=%.4f, acc(adjusted)=%.4f, err=%.2f%%, \nspd(max)=%.4f, spd(cont)=%.4f, spd(adjusted)=%.4f, err=%.2f%%\n',acc,acc_cont,accr,err_acc*100,spd,spd_cont,spdr,err_spd*100)

    if err_spd > 0.2 || err_acc >0.2
       disp('warning: trajectory deviates more than 20% from set values') 
    end
    
    % init speed vector
    y = zeros(1,n);
    y(1) = T0;
    
    % init acc vector (optional)
    ddy = zeros(1,n);
    
    for l=2:n
        y(l) = y(l-1) + ts*(dy(l)+dy(l-1))/2;   % Integration by trapezoidal rule
        ddy(l) = (dy(l)-dy(l-1))/ts;            % Forward difference quotients
    end
    
end